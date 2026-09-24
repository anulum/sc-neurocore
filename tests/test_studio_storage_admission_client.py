# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — bounded named storage admission client tests

"""Exercise the production sender against real peer-bound service preparation."""

from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import socket
import time
from collections.abc import Mapping
from typing import cast

import pytest

from sc_neurocore.studio.platform.policy_gateway import PolicyGateway
from sc_neurocore.studio.platform.policy_models import InMemoryAuditSink
from sc_neurocore.studio.platform.jobs_admission import StudioJobQueueFull
from sc_neurocore.studio.platform.jobs_ledger import StudioJobLedger
from sc_neurocore.studio.platform.jobs_shared_admission import SharedJobAdmission
from sc_neurocore.studio.platform.storage_admission_client import (
    PendingNamedAdmission,
    read_named_admission_result,
    send_named_admission_request,
)
from sc_neurocore.studio.platform.storage_admission_protocol import (
    StorageNamedAdmissionRequest,
    decode_named_admission_request,
)
from sc_neurocore.studio.platform.storage_admission_response import encode_named_admission_response
from sc_neurocore.studio.platform.storage_configuration import StorageBoundaryConfiguration
from sc_neurocore.studio.platform.storage_named_admission import prepare_named_admission
from sc_neurocore.studio.platform.storage_peer import write_verified_frame
from sc_neurocore.studio.platform.storage_record_protocol import StorageRequester


def _configuration(
    root: Path,
    *,
    metadata_limit: int = 4096,
    service_uid: int | None = None,
    timeout: float = 2.0,
) -> StorageBoundaryConfiguration:
    return StorageBoundaryConfiguration(
        storage_uid=os.getuid() if service_uid is None else service_uid,
        api_uid=os.getuid() + 1,
        worker_uid=os.getuid() + 2,
        authority_root=root / "authority",
        spool_root=root / "spool",
        socket_path=root / "endpoint" / "storage.sock",
        workspace="default",
        frame_max_bytes=4096,
        max_metadata_bytes=metadata_limit,
        max_seed_bytes=8,
        max_seed_entries=2,
        max_manifest_bytes=min(64, metadata_limit),
        max_artifact_bytes=65536,
        max_artifact_entries=16,
        transfer_timeout_seconds=timeout,
        max_connections=1,
    )


def _request(
    *, workspace: str = "default", manifest: dict[str, int] | None = None
) -> StorageNamedAdmissionRequest:
    return StorageNamedAdmissionRequest(
        schema_version="studio.storage.admission.v1",
        operation="admit_named",
        request_id="trace-one",
        mutation_id="retry-one",
        workspace=workspace,
        requester=StorageRequester(principal_id="operator", roles=("studio.admin",)),
        task_name="analysis.run",
        authorized_route="/api/analysis/jobs",
        payload={"model": "lif"},
        seed_manifest={"input/data.bin": 2} if manifest is None else manifest,
        execution_timeout_seconds=30.0,
        queue_wait_seconds=None,
        admission=None,
        training_config=None,
        experiment_sha256=None,
    )


def test_production_sender_reaches_service_policy_and_replay_preparation(tmp_path: Path) -> None:
    """The typed client and real service agree on exact metadata and seed bytes."""
    configuration = _configuration(tmp_path)
    reader, writer = socket.socketpair()
    with reader, writer:
        request = _request()
        expected = send_named_admission_request(
            writer,
            request=request,
            seed_inputs={"input/data.bin": b"ab"},
            configuration=configuration,
        )
        sink = InMemoryAuditSink()
        prepared = prepare_named_admission(
            reader,
            gateway=PolicyGateway(sink),
            workspace=configuration.workspace,
            expected_api_uid=os.getuid(),
            frame_max_bytes=configuration.frame_max_bytes,
            max_metadata_bytes=configuration.max_metadata_bytes,
            max_seed_bytes=configuration.max_seed_bytes,
            max_seed_entries=configuration.max_seed_entries,
            max_manifest_bytes=configuration.max_manifest_bytes,
            deadline=time.monotonic() + 2,
        )
        assert prepared.seed_inputs == (("input/data.bin", b"ab"),)
        assert prepared.replay == expected.replay
        assert prepared.request_json == expected.request_json
        assert len(sink.events) == 1
        assert sink.events[0].decision == "allow"


@pytest.mark.parametrize(
    "manifest,seeds",
    [
        ({"input/data.bin": 2}, {"input/data.bin": b"x"}),
        ({"../bad": 1}, {"../bad": b"x"}),
        ({"input/data.bin": 2}, {"input/data.bin": b"ab", "other": b""}),
    ],
)
def test_invalid_seed_declaration_sends_no_metadata(
    tmp_path: Path, manifest: dict[str, int], seeds: dict[str, bytes]
) -> None:
    """An inconsistent transfer never sends even the metadata frame header."""
    reader, writer = socket.socketpair()
    with reader, writer:
        with pytest.raises(ValueError):
            send_named_admission_request(
                writer,
                request=_request(manifest=manifest),
                seed_inputs=seeds,
                configuration=_configuration(tmp_path),
            )
        assert writer.fileno() == -1
        assert reader.recv(1) == b""


def test_correlated_result_uses_sent_snapshot_after_caller_mutates_request(
    tmp_path: Path,
) -> None:
    """A changed nested payload cannot retarget a reply after wire transfer."""
    configuration = _configuration(tmp_path)
    reader, writer = socket.socketpair()
    ledger = StudioJobLedger(root=tmp_path / "ledger")
    try:
        with reader, writer:
            request = _request()
            pending = send_named_admission_request(
                writer,
                request=request,
                seed_inputs={"input/data.bin": b"ab"},
                configuration=configuration,
            )
            prepared = prepare_named_admission(
                reader,
                gateway=PolicyGateway(InMemoryAuditSink()),
                workspace=configuration.workspace,
                expected_api_uid=os.getuid(),
                frame_max_bytes=configuration.frame_max_bytes,
                max_metadata_bytes=configuration.max_metadata_bytes,
                max_seed_bytes=configuration.max_seed_bytes,
                max_seed_entries=configuration.max_seed_entries,
                max_manifest_bytes=configuration.max_manifest_bytes,
                deadline=pending.deadline,
            )
            request.payload["model"] = "changed"
            original = decode_named_admission_request(
                pending.request_json, max_metadata_bytes=len(pending.request_json)
            )
            assert original.payload == {"model": "lif"}
            admission = SharedJobAdmission(ledger, max_concurrent=1, max_queued=0)
            outcome = admission.admit(
                job_id="sj_0000000000000001",
                kind=prepared.task.kind,
                actor=prepared.task.owner,
                workspace=configuration.workspace,
                request_id=original.request_id,
                idempotency_key=None,
                experiment_sha256=None,
                admission={},
                execution_model="process",
                replay=prepared.replay,
                supervisor=prepared.supervisor,
            )
            response = encode_named_admission_response(
                request=original,
                replay=prepared.replay,
                outcome=outcome,
                max_bytes=configuration.frame_max_bytes,
            )
            write_verified_frame(
                reader,
                response,
                expected_uid=os.getuid(),
                max_bytes=configuration.frame_max_bytes,
                deadline=pending.deadline,
            )
            assert read_named_admission_result(writer, pending=pending) == outcome.record.job_id
            assert writer.fileno() == -1
    finally:
        ledger.close()


def test_uncorrelated_result_closes_client_channel(tmp_path: Path) -> None:
    """A verified service peer still cannot substitute another trace result."""
    configuration = _configuration(tmp_path)
    reader, writer = socket.socketpair()
    with reader, writer:
        pending = send_named_admission_request(
            writer,
            request=_request(),
            seed_inputs={"input/data.bin": b"ab"},
            configuration=configuration,
        )
        response = {
            "schema_version": "studio.storage.admission.v1",
            "operation": "admit_named_result",
            "request_id": "other",
            "mutation_id": "retry-one",
            "workspace": "default",
            "task_name": "analysis.run",
            "payload_sha256": pending.replay.payload_sha256,
            "status": "admitted",
            "job_id": "sj_0000000000000001",
            "running": None,
            "queued": None,
            "limit": None,
        }
        write_verified_frame(
            reader,
            json.dumps(response).encode(),
            expected_uid=os.getuid(),
            max_bytes=configuration.frame_max_bytes,
            deadline=pending.deadline,
        )
        with pytest.raises(ValueError, match="does not match"):
            read_named_admission_result(writer, pending=pending)
        assert writer.fileno() == -1


def test_lost_reply_obeys_original_deadline_and_closes(tmp_path: Path) -> None:
    """An absent response is ambiguous and never renews the transfer timer."""
    configuration = _configuration(tmp_path, timeout=0.1)
    reader, writer = socket.socketpair()
    with reader, writer:
        pending = send_named_admission_request(
            writer,
            request=_request(),
            seed_inputs={"input/data.bin": b"ab"},
            configuration=configuration,
        )
        with pytest.raises(TimeoutError):
            read_named_admission_result(writer, pending=pending)
        assert writer.fileno() == -1


def test_correlated_queue_refusal_crosses_verified_reply_frame(tmp_path: Path) -> None:
    """A capacity refusal reaches the client with exact counts and closes."""
    configuration = _configuration(tmp_path)
    reader, writer = socket.socketpair()
    with reader, writer:
        pending = send_named_admission_request(
            writer,
            request=_request(),
            seed_inputs={"input/data.bin": b"ab"},
            configuration=configuration,
        )
        original = decode_named_admission_request(
            pending.request_json, max_metadata_bytes=len(pending.request_json)
        )
        response = encode_named_admission_response(
            request=original,
            replay=pending.replay,
            outcome=StudioJobQueueFull(running=1, queued=0, limit=0),
            max_bytes=configuration.frame_max_bytes,
        )
        write_verified_frame(
            reader,
            response,
            expected_uid=os.getuid(),
            max_bytes=configuration.frame_max_bytes,
            deadline=pending.deadline,
        )
        with pytest.raises(StudioJobQueueFull) as refused:
            read_named_admission_result(writer, pending=pending)
        assert (refused.value.running, refused.value.queued, refused.value.limit) == (1, 0, 0)
        assert writer.fileno() == -1


def test_reader_refuses_fabricated_pending_state(tmp_path: Path) -> None:
    """Only one valid sent snapshot may be used to interpret a response."""
    configuration = _configuration(tmp_path)
    reader, writer = socket.socketpair()
    with reader, writer:
        pending = send_named_admission_request(
            writer,
            request=_request(),
            seed_inputs={"input/data.bin": b"ab"},
            configuration=configuration,
        )
        with pytest.raises(ValueError, match="pending"):
            read_named_admission_result(writer, pending=cast(PendingNamedAdmission, object()))
        assert writer.fileno() == -1
        assert reader.recv(1)

    reader, writer = socket.socketpair()
    with reader, writer:
        pending = send_named_admission_request(
            writer,
            request=_request(),
            seed_inputs={"input/data.bin": b"ab"},
            configuration=configuration,
        )
        with pytest.raises(PermissionError):
            read_named_admission_result(
                writer, pending=replace(pending, service_uid=os.getuid() + 3)
            )
        assert writer.fileno() == -1


def test_workspace_or_metadata_limit_refuses_before_transfer(tmp_path: Path) -> None:
    """Client configuration, not request content, controls scope and byte budget."""
    for request, configuration in (
        (_request(workspace="other"), _configuration(tmp_path)),
        (_request(), _configuration(tmp_path, metadata_limit=64)),
    ):
        reader, writer = socket.socketpair()
        with reader, writer:
            with pytest.raises(ValueError):
                send_named_admission_request(
                    writer,
                    request=request,
                    seed_inputs={"input/data.bin": b"ab"},
                    configuration=configuration,
                )
            assert writer.fileno() == -1
            assert reader.recv(1) == b""


def test_wrong_service_uid_sends_no_frame(tmp_path: Path) -> None:
    """The actual connected peer must match the configured service identity."""
    configuration = _configuration(tmp_path, service_uid=os.getuid() + 3)
    reader, writer = socket.socketpair()
    with reader, writer:
        with pytest.raises(PermissionError):
            send_named_admission_request(
                writer,
                request=_request(),
                seed_inputs={"input/data.bin": b"ab"},
                configuration=configuration,
            )
        assert writer.fileno() == -1
        assert reader.recv(1) == b""


@pytest.mark.parametrize(
    "admission_request,seeds",
    [
        (cast(StorageNamedAdmissionRequest, object()), {"input/data.bin": b"ab"}),
        (_request(), cast(Mapping[str, bytes], [])),
        (_request(), cast(Mapping[str, bytes], {1: b"ab"})),
        (_request(), {"input/data.bin": cast(bytes, bytearray(b"ab"))}),
    ],
)
def test_invalid_client_values_close_without_emitting_metadata(
    tmp_path: Path,
    admission_request: StorageNamedAdmissionRequest,
    seeds: Mapping[str, bytes],
) -> None:
    """Only a typed request and immutable named bytes can reach the authority."""
    reader, writer = socket.socketpair()
    with reader, writer:
        with pytest.raises(ValueError):
            send_named_admission_request(
                writer,
                request=admission_request,
                seed_inputs=seeds,
                configuration=_configuration(tmp_path),
            )
        assert writer.fileno() == -1
        assert reader.recv(1) == b""


@pytest.mark.parametrize(
    "change",
    [
        {"requester": None},
        {"task_name": "unknown"},
        {"authorized_route": "/api/models/scan/jobs"},
    ],
)
def test_unauthenticated_or_unreviewed_intent_never_reaches_service(
    tmp_path: Path, change: dict[str, object]
) -> None:
    """The client refuses an absent claim or unreviewed task/route pair."""
    admission_request = _request().model_copy(update=change)
    reader, writer = socket.socketpair()
    with reader, writer:
        with pytest.raises(ValueError):
            send_named_admission_request(
                writer,
                request=admission_request,
                seed_inputs={"input/data.bin": b"ab"},
                configuration=_configuration(tmp_path),
            )
        assert writer.fileno() == -1
        assert reader.recv(1) == b""
