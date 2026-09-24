# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — peer-bound named admission preparation tests

"""Exercise real framed, peer-bound policy and seed preparation."""

from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import time

import pytest

from sc_neurocore.studio.platform.jobs_ledger_supervisor import supervisor_identity
from sc_neurocore.studio.platform.policy_gateway import PolicyGateway
from sc_neurocore.studio.platform.policy_models import InMemoryAuditSink
from sc_neurocore.studio.platform.storage_admission_protocol import decode_named_admission_request
from sc_neurocore.studio.platform.storage_named_admission import (
    PreparedNamedAdmission,
    prepare_named_admission,
)
from sc_neurocore.studio.platform.storage_peer import read_verified_frame, write_verified_frame
from sc_neurocore.studio.platform.storage_seed_ingress import send_storage_seeds


def _request() -> dict[str, object]:
    return {
        "schema_version": "studio.storage.admission.v1",
        "operation": "admit_named",
        "request_id": "trace-one",
        "mutation_id": "retry-one",
        "workspace": "default",
        "requester": {"principal_id": "operator", "roles": ["studio.admin"]},
        "task_name": "analysis.run",
        "authorized_route": "/api/analysis/jobs",
        "payload": {"model": "lif"},
        "seed_manifest": {"input/data.bin": 2},
        "execution_timeout_seconds": 30.0,
        "queue_wait_seconds": None,
        "admission": None,
        "training_config": None,
        "experiment_sha256": None,
    }


def _exchange(
    request: dict[str, object],
    seeds: dict[str, bytes],
    *,
    sink: InMemoryAuditSink,
    workspace: str = "default",
    pre_read_metadata: bool = False,
    authority_dirfd: int | None = None,
) -> PreparedNamedAdmission:
    reader, writer = socket.socketpair()
    with reader, writer:
        deadline = time.monotonic() + 2
        write_verified_frame(
            writer,
            json.dumps(request).encode("utf-8"),
            expected_uid=os.getuid(),
            max_bytes=4096,
            deadline=deadline,
        )
        manifest = request["seed_manifest"]
        assert isinstance(manifest, dict)
        send_storage_seeds(
            writer,
            seed_inputs=seeds,
            manifest=manifest,
            expected_service_uid=os.getuid(),
            frame_max_bytes=4096,
            max_seed_bytes=8,
            max_seed_entries=2,
            max_manifest_bytes=64,
            deadline=deadline,
        )
        initial_frame = (
            read_verified_frame(
                reader,
                expected_uid=os.getuid(),
                max_bytes=4096,
                deadline=deadline,
            )
            if pre_read_metadata
            else None
        )
        prepared = prepare_named_admission(
            reader,
            gateway=PolicyGateway(sink),
            workspace=workspace,
            expected_api_uid=os.getuid(),
            frame_max_bytes=4096,
            max_metadata_bytes=4096,
            max_seed_bytes=8,
            max_seed_entries=2,
            max_manifest_bytes=64,
            deadline=deadline,
            initial_frame=initial_frame,
            authority_dirfd=authority_dirfd,
        )
        assert prepared.supervisor == supervisor_identity()
        assert prepared.task.name == "analysis.run"
        assert prepared.requester.principal_id == "operator"
        return prepared


def test_real_verified_metadata_and_seed_content_prepare_one_named_intent() -> None:
    """Policy audit and replay bind actual bytes without creating a ledger job."""
    sink = InMemoryAuditSink()
    prepared = _exchange(_request(), {"input/data.bin": b"ab"}, sink=sink)
    assert len(prepared.replay.payload_sha256) == 64
    assert prepared.seed_inputs == (("input/data.bin", b"ab"),)
    restored = decode_named_admission_request(
        prepared.request_json, max_metadata_bytes=len(prepared.request_json)
    )
    assert restored.payload == {"model": "lif"}
    restored.payload["model"] = "changed"
    assert decode_named_admission_request(
        prepared.request_json, max_metadata_bytes=len(prepared.request_json)
    ).payload == {"model": "lif"}
    assert len(sink.events) == 1
    assert sink.events[0].decision == "allow"
    assert sink.events[0].route == "/api/analysis/jobs"


def test_file_backed_prepare_preserves_replay_and_handle_lifetime(tmp_path: Path) -> None:
    """Policy-approved named admission can retain only unlinked file handles."""
    direct = _exchange(_request(), {"input/data.bin": b"ab"}, sink=InMemoryAuditSink())
    rootfd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        prepared = _exchange(
            _request(),
            {"input/data.bin": b"ab"},
            sink=InMemoryAuditSink(),
            authority_dirfd=rootfd,
        )
        assert prepared.replay == direct.replay
        assert prepared.seed_inputs == ()
        staged = prepared.seed_files
        assert staged is not None
        with staged:
            assert len(staged.files) == 1
            assert staged.files[0].stream.read() == b"ab"
            assert os.fstat(staged.files[0].stream.fileno()).st_nlink == 0
        assert staged.files[0].stream.closed
        assert list(tmp_path.iterdir()) == []
    finally:
        os.close(rootfd)


def test_listener_pre_read_metadata_preserves_named_seed_and_replay_identity() -> None:
    """Dispatch can hand the first verified frame to admission without losing seeds."""
    request = _request()
    direct = _exchange(request, {"input/data.bin": b"ab"}, sink=InMemoryAuditSink())
    dispatched = _exchange(
        request,
        {"input/data.bin": b"ab"},
        sink=InMemoryAuditSink(),
        pre_read_metadata=True,
    )
    assert dispatched.seed_inputs == direct.seed_inputs
    assert dispatched.replay == direct.replay


def test_changed_received_seed_bytes_change_replay_digest() -> None:
    """One mutation ID cannot silently replay a different on-wire seed."""
    first = _exchange(_request(), {"input/data.bin": b"ab"}, sink=InMemoryAuditSink())
    changed = _exchange(_request(), {"input/data.bin": b"ac"}, sink=InMemoryAuditSink())
    assert first.replay.payload_sha256 != changed.replay.payload_sha256


def test_service_requires_configured_workspace_before_peer_transfer() -> None:
    """An absent server workspace never accepts a client-selected scope."""
    reader, writer = socket.socketpair()
    with reader, writer, pytest.raises(ValueError, match="workspace"):
        prepare_named_admission(
            reader,
            gateway=PolicyGateway(InMemoryAuditSink()),
            workspace="",
            expected_api_uid=os.getuid(),
            frame_max_bytes=4096,
            max_metadata_bytes=4096,
            max_seed_bytes=8,
            max_seed_entries=2,
            max_manifest_bytes=64,
            deadline=time.monotonic() + 1,
        )


@pytest.mark.parametrize(
    "change,workspace",
    [
        ({"workspace": "other"}, "default"),
        ({"task_name": "unknown"}, "default"),
        ({"authorized_route": "/api/models/scan/jobs"}, "default"),
        ({"requester": None}, "default"),
        (
            {
                "task_name": "audit.quarantine_archive",
                "authorized_route": "/api/studio/audit/quarantine/archive",
                "requester": {"principal_id": "reader", "roles": []},
            },
            "default",
        ),
    ],
)
def test_invalid_scope_task_or_principal_refuses_before_seed_read(
    change: dict[str, object], workspace: str
) -> None:
    """Workspace, registry and policy decisions precede seed transfer."""
    request = _request()
    request.update(change)
    reader, writer = socket.socketpair()
    sink = InMemoryAuditSink()
    with reader, writer:
        deadline = time.monotonic() + 1
        write_verified_frame(
            writer,
            json.dumps(request).encode("utf-8"),
            expected_uid=os.getuid(),
            max_bytes=4096,
            deadline=deadline,
        )
        with pytest.raises((ValueError, PermissionError)):
            prepare_named_admission(
                reader,
                gateway=PolicyGateway(sink),
                workspace=workspace,
                expected_api_uid=os.getuid(),
                frame_max_bytes=4096,
                max_metadata_bytes=4096,
                max_seed_bytes=8,
                max_seed_entries=2,
                max_manifest_bytes=64,
                deadline=deadline,
            )
        assert reader.fileno() == -1
        if request["requester"] is None or request["requester"] == {
            "principal_id": "reader",
            "roles": [],
        }:
            assert len(sink.events) == 1
            assert sink.events[0].decision == "deny"
        else:
            assert not sink.events
