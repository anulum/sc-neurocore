# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — correlated named admission response tests

"""Exercise durable SQLite admission outcomes through exact framed responses."""

from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import socket
import time
from typing import cast

import pytest

from sc_neurocore.studio.platform.jobs_admission import StudioJobQueueFull
from sc_neurocore.studio.platform.jobs_admission_replay import StorageAdmissionReplay
from sc_neurocore.studio.platform.jobs_ledger import StudioJobLedger
from sc_neurocore.studio.platform.jobs_ledger_schema import StudioJobSubmission
from sc_neurocore.studio.platform.jobs_shared_admission import SharedJobAdmission
from sc_neurocore.studio.platform.storage_admission_protocol import StorageNamedAdmissionRequest
from sc_neurocore.studio.platform.storage_admission_response import (
    decode_named_admission_response,
    encode_named_admission_response,
)
from sc_neurocore.studio.platform.storage_peer import read_verified_frame, write_verified_frame
from sc_neurocore.studio.platform.storage_record_protocol import StorageRequester


def _request(
    *, request_id: str = "trace-one", mutation_id: str = "retry-one"
) -> StorageNamedAdmissionRequest:
    return StorageNamedAdmissionRequest(
        schema_version="studio.storage.admission.v1",
        operation="admit_named",
        request_id=request_id,
        mutation_id=mutation_id,
        workspace="default",
        requester=StorageRequester(principal_id="operator", roles=("studio.admin",)),
        task_name="analysis.run",
        authorized_route="/api/analysis/jobs",
        payload={"model": "lif"},
        seed_manifest={},
        execution_timeout_seconds=30.0,
        queue_wait_seconds=None,
        admission=None,
        training_config=None,
        experiment_sha256=None,
    )


def _admit(
    admission: SharedJobAdmission,
    *,
    job_id: str,
    replay: StorageAdmissionReplay,
    request_id: str = "trace-one",
) -> StudioJobSubmission:
    return admission.admit(
        job_id=job_id,
        kind="analysis",
        actor="studio",
        workspace="default",
        request_id=request_id,
        idempotency_key=None,
        experiment_sha256=None,
        admission={},
        execution_model="process",
        replay=replay,
    )


@pytest.fixture
def admitted(
    tmp_path: Path,
) -> tuple[StorageNamedAdmissionRequest, StorageAdmissionReplay, StudioJobSubmission]:
    """Return one real SQLite-admitted named job and its replay identity."""
    request = _request()
    replay = StorageAdmissionReplay("operator", "retry-one", "a" * 64)
    ledger = StudioJobLedger(root=tmp_path)
    try:
        admission = SharedJobAdmission(ledger, max_concurrent=1, max_queued=0)
        outcome = _admit(admission, job_id="sj_0000000000000001", replay=replay)
        return request, replay, outcome
    finally:
        ledger.close()


def test_real_admission_result_survives_peer_verified_frame(
    admitted: tuple[StorageNamedAdmissionRequest, StorageAdmissionReplay, StudioJobSubmission],
) -> None:
    """Client receives the exact committed job ID after versioned correlation."""
    request, replay, outcome = admitted
    payload = encode_named_admission_response(
        request=request, replay=replay, outcome=outcome, max_bytes=8192
    )
    reader, writer = socket.socketpair()
    with reader, writer:
        deadline = time.monotonic() + 2
        write_verified_frame(
            writer, payload, expected_uid=os.getuid(), max_bytes=8192, deadline=deadline
        )
        received = read_verified_frame(
            reader, expected_uid=os.getuid(), max_bytes=8192, deadline=deadline
        )
        assert (
            decode_named_admission_response(
                received, request=request, replay=replay, max_bytes=8192
            )
            == outcome.record.job_id
        )


def test_lost_reply_replays_same_job_with_new_http_trace(tmp_path: Path) -> None:
    """Durable mutation replay returns the same job ID after ledger restart."""
    replay = StorageAdmissionReplay("operator", "retry-one", "a" * 64)
    first_ledger = StudioJobLedger(root=tmp_path)
    first_admission = SharedJobAdmission(first_ledger, max_concurrent=1, max_queued=0)
    original = _admit(first_admission, job_id="sj_0000000000000001", replay=replay)
    first_ledger.close()
    second_ledger = StudioJobLedger(root=tmp_path)
    try:
        second_admission = SharedJobAdmission(second_ledger, max_concurrent=1, max_queued=0)
        repeated = _admit(
            second_admission,
            job_id="sj_0000000000000002",
            replay=replay,
            request_id="trace-retry",
        )
        assert repeated.duplicate
        request = _request(request_id="trace-retry")
        payload = encode_named_admission_response(
            request=request, replay=replay, outcome=repeated, max_bytes=8192
        )
        assert (
            decode_named_admission_response(payload, request=request, replay=replay, max_bytes=8192)
            == original.record.job_id
        )
        assert second_admission.snapshot().admitted == 1
    finally:
        second_ledger.close()


def test_real_capacity_refusal_has_exact_correlated_counts(tmp_path: Path) -> None:
    """A full SQLite queue yields a bounded refusal with durable dimensions."""
    replay = StorageAdmissionReplay("operator", "retry-one", "a" * 64)
    ledger = StudioJobLedger(root=tmp_path)
    try:
        admission = SharedJobAdmission(ledger, max_concurrent=1, max_queued=0)
        _admit(admission, job_id="sj_0000000000000001", replay=replay)
        refusal_replay = StorageAdmissionReplay("operator", "retry-two", "b" * 64)
        with pytest.raises(StudioJobQueueFull) as captured:
            _admit(admission, job_id="sj_0000000000000002", replay=refusal_replay)
        request = _request(mutation_id="retry-two")
        payload = encode_named_admission_response(
            request=request, replay=refusal_replay, outcome=captured.value, max_bytes=8192
        )
        with pytest.raises(StudioJobQueueFull) as decoded:
            decode_named_admission_response(
                payload, request=request, replay=refusal_replay, max_bytes=8192
            )
        assert (decoded.value.running, decoded.value.queued, decoded.value.limit) == (1, 0, 0)
        response = json.loads(payload)
        response["job_id"] = "sj_0000000000000001"
        with pytest.raises(ValueError, match="queue-full"):
            decode_named_admission_response(
                json.dumps(response).encode(),
                request=request,
                replay=refusal_replay,
                max_bytes=8192,
            )
    finally:
        ledger.close()


@pytest.mark.parametrize(
    "field,value",
    [
        ("request_id", "other"),
        ("mutation_id", "other"),
        ("workspace", "other"),
        ("task_name", "model.scan"),
        ("payload_sha256", "b" * 64),
        ("status", "prepared"),
        ("job_id", None),
        ("running", 1),
        ("extra", "bad"),
    ],
)
def test_changed_response_refuses_without_job_id(
    admitted: tuple[StorageNamedAdmissionRequest, StorageAdmissionReplay, StudioJobSubmission],
    field: str,
    value: object,
) -> None:
    """No schema, correlation or outcome-shape error becomes a job ID."""
    request, replay, outcome = admitted
    response = json.loads(
        encode_named_admission_response(
            request=request, replay=replay, outcome=outcome, max_bytes=8192
        )
    )
    response[field] = value
    with pytest.raises(ValueError):
        decode_named_admission_response(
            json.dumps(response).encode(), request=request, replay=replay, max_bytes=8192
        )


@pytest.mark.parametrize(
    "payload",
    [
        b'{"status":"admitted","status":"queue_full"}',
        b'{"payload_sha256":NaN}',
        b"\xff",
        b"{",
        b"",
    ],
)
def test_malformed_response_refuses(
    payload: bytes,
    admitted: tuple[StorageNamedAdmissionRequest, StorageAdmissionReplay, StudioJobSubmission],
) -> None:
    """Duplicate names, nonfinite constants, bad JSON and empty frames fail closed."""
    request, replay, _ = admitted
    with pytest.raises(ValueError):
        decode_named_admission_response(payload, request=request, replay=replay, max_bytes=8192)


def test_encoder_refuses_wrong_domain_record_and_frame_budget(
    admitted: tuple[StorageNamedAdmissionRequest, StorageAdmissionReplay, StudioJobSubmission],
) -> None:
    """No mismatched owner, workspace or oversized outcome leaves the service."""
    request, replay, outcome = admitted
    wrong = StudioJobSubmission(replace(outcome.record, owner="other"), duplicate=False)
    with pytest.raises(ValueError, match="named request"):
        encode_named_admission_response(
            request=request, replay=replay, outcome=wrong, max_bytes=8192
        )
    with pytest.raises(ValueError, match="frame limit"):
        encode_named_admission_response(
            request=request, replay=replay, outcome=outcome, max_bytes=8
        )
    with pytest.raises(ValueError, match="replay"):
        encode_named_admission_response(
            request=request,
            replay=StorageAdmissionReplay("other", "retry-one", "a" * 64),
            outcome=outcome,
            max_bytes=8192,
        )
    with pytest.raises(ValueError, match="response limit"):
        encode_named_admission_response(
            request=request, replay=replay, outcome=outcome, max_bytes=0
        )
    with pytest.raises(ValueError, match="unsupported"):
        encode_named_admission_response(
            request=request,
            replay=replay,
            outcome=cast(StudioJobSubmission | StudioJobQueueFull, object()),
            max_bytes=8192,
        )
    with pytest.raises(ValueError, match="correlation"):
        decode_named_admission_response(
            b"{}",
            request=cast(StorageNamedAdmissionRequest, object()),
            replay=replay,
            max_bytes=8192,
        )
