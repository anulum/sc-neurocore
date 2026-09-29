# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Event contract bulk admission acceptance

"""Admit full recording declarations through real sockets and durable replay."""

import hashlib
import json
import os
import time
from pathlib import Path

import pytest

from sc_neurocore.studio.platform.jobs_ledger import StudioJobLedger
from sc_neurocore.studio.platform.storage_admission_client import (
    send_named_admission_request,
    read_named_admission_result,
)
from sc_neurocore.studio.platform.storage_admission_protocol import StorageNamedAdmissionRequest
from sc_neurocore.studio.platform.storage_record_protocol import StorageRequester
from sc_neurocore.studio.platform.storage_event_admission import (
    encode_event_admission,
    send_event_admission_content,
)
from sc_neurocore.studio.platform.storage_peer import write_verified_frame, read_verified_frame
from tests.studio_storage_generation_runs import (
    base as base,
    ledger as ledger,
    authority as authority,
)
from tests.studio_storage_generation_support import Authority, FRAME
from tests.test_studio_storage_isolated_jobs import _configuration as boundary
from tests.test_studio_training_config_storage import _configuration as recording_configuration


def _request(
    config: dict[str, object], *, roles: tuple[str, ...] = ("studio.admin",)
) -> StorageNamedAdmissionRequest:
    """Declare a real training snapshot under the reviewed named task."""
    return StorageNamedAdmissionRequest(
        schema_version="studio.storage.admission.v1",
        operation="admit_named",
        request_id="event-trace",
        mutation_id="event-retry",
        workspace="default",
        requester=StorageRequester(principal_id="operator", roles=roles),
        task_name="training.start",
        authorized_route="/api/training/start",
        payload=config,
        training_config=config,
        seed_manifest={},
        execution_timeout_seconds=120.0,
        queue_wait_seconds=None,
        admission=None,
        experiment_sha256=None,
    )


@pytest.mark.parametrize("wrapped", [False, True])
def test_full_manifest_admitted_and_replayed_under_original_frame_budget(
    base: Path,
    tmp_path: Path,
    authority: Authority,
    ledger: StudioJobLedger,
    wrapped: bool,
) -> None:
    """A thousand real SHD samples retain full metadata and one durable job."""
    config = recording_configuration(tmp_path / "recordings", count=1000)
    assert len(json.dumps(config).encode()) > FRAME
    request = _request(config)
    if wrapped:
        request = request.model_copy(update={"payload": {"config": config}})
    jobs = []
    for _ in range(2):
        channel = authority.connect()
        pending = send_named_admission_request(
            channel, request=request, seed_inputs={}, configuration=boundary(base)
        )
        jobs.append(read_named_admission_result(channel, pending=pending))
    assert jobs[0] == jobs[1]
    assert ledger.record(jobs[0]).training_config == config
    assert len(ledger.list_records()) == 1
    assert authority.failures == []
    assert authority.services.frame_max_bytes == FRAME
    assert authority.services.max_metadata_bytes == FRAME


@pytest.mark.parametrize(
    "failure",
    ["sha", "chunk", "denied", "reference", "snapshot", "process", "duplicate", "metadata"],
)
def test_invalid_event_transfer_creates_no_job(
    base: Path,
    tmp_path: Path,
    ledger: StudioJobLedger,
    failure: str,
) -> None:
    """Actual corrupt frames and unauthorized transfers leave the ledger empty."""
    config = recording_configuration(tmp_path / "recordings", count=1000)
    authority = Authority(ledger)
    request = _request(config)
    if failure == "denied":
        request = request.model_copy(update={"requester": None})
    metadata, events = encode_event_admission(request, max_metadata_bytes=FRAME)
    assert events is not None and len(events) > FRAME
    if failure in {"snapshot", "process"}:
        parsed = json.loads(metadata)
        if failure == "snapshot":
            parsed["request"]["training_config"] = None
        else:
            parsed["request"]["payload"] = {"config": 1}
        metadata = json.dumps(parsed).encode()
    if failure == "duplicate":
        metadata = b'{"operation":"admit_named",' + metadata[1:]
    if failure == "reference":
        events = b" " + events[1:]
        parsed = json.loads(metadata)
        parsed["event_sha256"] = hashlib.sha256(events).hexdigest()
        metadata = json.dumps(parsed).encode()
    if failure == "metadata":
        from dataclasses import replace

        authority.services = replace(authority.services, max_metadata_bytes=32)
    channel = authority.connect()
    with channel:
        deadline = time.monotonic() + 10
        write_verified_frame(
            channel, metadata, expected_uid=os.getuid(), max_bytes=FRAME, deadline=deadline
        )
        if failure == "chunk":
            write_verified_frame(
                channel, b"short", expected_uid=os.getuid(), max_bytes=FRAME, deadline=deadline
            )
        elif failure in {"sha", "reference", "snapshot", "process"}:
            send_event_admission_content(
                channel,
                b"!" + events[1:] if failure == "sha" else events,
                expected_uid=os.getuid(),
                frame_max_bytes=FRAME,
                deadline=deadline,
            )
        with pytest.raises(EOFError):
            read_verified_frame(
                channel, expected_uid=os.getuid(), max_bytes=FRAME, deadline=deadline
            )
    assert ledger.list_records() == ()
    authority.join(expected=(PermissionError if failure == "denied" else ValueError,))


@pytest.mark.parametrize("failure", ["snapshot", "metadata"])
def test_sender_refuses_invalid_event_intent_without_transmitting(
    base: Path,
    tmp_path: Path,
    failure: str,
) -> None:
    """Bad snapshots and oversized headers close the real stream before any bytes."""
    import socket

    config = recording_configuration(tmp_path / "recordings", count=1000)
    request = _request(config)
    configuration = boundary(base)
    if failure == "snapshot":
        request = request.model_copy(update={"payload": {**config, "seed": 88}})
    else:
        configuration = configuration.model_copy(
            update={"max_metadata_bytes": 32, "max_manifest_bytes": 32}
        )
    reader, writer = socket.socketpair()
    with reader, writer:
        with pytest.raises(ValueError):
            send_named_admission_request(
                writer, request=request, seed_inputs={}, configuration=configuration
            )
        assert reader.recv(1) == b""


@pytest.mark.parametrize(
    "metadata",
    [b'{"operation":"admit_named","operation":"admit_named"}', b'{"payload":{"value":NaN}}'],
)
def test_direct_preparation_rejects_ambiguous_envelope_fields(base: Path, metadata: bytes) -> None:
    """The public direct receiver refuses ambiguous metadata without a listener."""
    import socket
    from sc_neurocore.studio.platform.storage_named_admission import prepare_named_admission
    from sc_neurocore.studio.platform.policy_gateway import PolicyGateway
    from sc_neurocore.studio.platform.policy_models import InMemoryAuditSink

    reader, writer = socket.socketpair()
    with reader, writer:
        deadline = time.monotonic() + 10
        write_verified_frame(
            writer,
            metadata,
            expected_uid=os.getuid(),
            max_bytes=FRAME,
            deadline=deadline,
        )
        config = boundary(base)
        with pytest.raises(ValueError, match="duplicate|nonfinite"):
            prepare_named_admission(
                reader,
                gateway=PolicyGateway(InMemoryAuditSink()),
                workspace=config.workspace,
                expected_api_uid=os.getuid(),
                frame_max_bytes=config.frame_max_bytes,
                max_metadata_bytes=config.max_metadata_bytes,
                max_seed_bytes=config.max_seed_bytes,
                max_seed_entries=config.max_seed_entries,
                max_manifest_bytes=config.max_manifest_bytes,
                deadline=deadline,
            )


def test_small_event_intent_retains_legacy_replay_digest(base: Path, tmp_path: Path) -> None:
    """The new sender preserves the exact published inline content identity."""
    import socket

    config = recording_configuration(tmp_path / "recordings", count=12)
    request = _request(config)
    legacy = {
        "schema_version": "studio.storage.admission-content.v1",
        "requester": {"principal_id": "operator", "roles": ["studio.admin"]},
        "workspace": "default",
        "kind": "training",
        "owner": "studio-training",
        "named_task": "training.start",
        "task_path": "sc_neurocore.studio.platform.training_process:run_training_process_task",
        "authorized_route": "/api/training/start",
        "payload": config,
        "seeds": [],
        "execution_timeout_seconds": 120.0,
        "queue_wait_seconds": None,
        "admission": None,
        "training_config": config,
        "experiment_sha256": None,
    }
    expected = json.dumps(legacy, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    assert len(expected) <= FRAME
    reader, writer = socket.socketpair()
    with reader, writer:
        pending = send_named_admission_request(
            writer, request=request, seed_inputs={}, configuration=boundary(base)
        )
    assert pending.replay.payload_sha256 == hashlib.sha256(expected).hexdigest()
