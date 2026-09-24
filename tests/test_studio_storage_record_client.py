# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Connected storage client acceptance

"""Exercise actual framed client/authority reads and refuse corrupt responses."""

import json
import os
import socket
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from sc_neurocore.studio.platform.jobs_ledger import StudioJobLedger
from sc_neurocore.studio.platform.jobs_models import StudioJobRecord
from sc_neurocore.studio.platform.policy_gateway import PolicyGateway
from sc_neurocore.studio.platform.policy_models import InMemoryAuditSink
from sc_neurocore.studio.platform.storage_peer import read_verified_frame, write_verified_frame
from sc_neurocore.studio.platform.storage_record import serve_record_read
from sc_neurocore.studio.platform.storage_record_client import (
    read_storage_record,
    read_storage_record_at_endpoint,
)
from sc_neurocore.studio.platform.storage_record_protocol import decode_record_request
from tests.studio_storage_listener_support import boundary


def _request(roles: list[str] | None = None, *, missing: bool = False) -> bytes:
    return json.dumps(
        {
            "schema_version": "studio.storage.record.v2",
            "operation": "record",
            "request_id": "client-trace",
            "job_id": "sj_0000000000000002" if missing else "sj_0000000000000001",
            "workspace": "default",
            "requester": None if roles is None else {"principal_id": "operator", "roles": roles},
        }
    ).encode()


@pytest.mark.parametrize("outcome", ["ok", "anonymous", "viewer", "missing"])
def test_client_reads_through_real_authority_without_changing_custody(
    tmp_path: Path, outcome: str
) -> None:
    """Real policy and SQLite retain cross-owner access and explicit denial."""
    ledger = StudioJobLedger(root=tmp_path / "authority")
    job_id = "sj_0000000000000001"
    ledger.create(
        job_id=job_id,
        kind="analysis",
        actor="service-owner",
        workspace="default",
        request_id="original-trace",
        idempotency_key="retained-key",
        experiment_sha256="a" * 64,
        admission={"budget": 3},
        execution_model="process",
    )
    before = ledger.record(job_id)
    history = ledger.transitions(job_id)
    roles = (
        None
        if outcome == "anonymous"
        else ["studio.viewer" if outcome == "viewer" else "studio.admin"]
    )
    request = decode_record_request(_request(roles, missing=outcome == "missing"))
    left, right = socket.socketpair()
    sink = InMemoryAuditSink()
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                serve_record_read,
                left,
                ledger=ledger,
                gateway=PolicyGateway(sink),
                workspace="default",
                expected_api_uid=os.getuid(),
                max_bytes=8192,
                deadline=time.monotonic() + 3,
            )
            try:
                if outcome == "ok":
                    result = read_storage_record(
                        right,
                        request=request,
                        expected_service_uid=os.getuid(),
                        max_bytes=8192,
                        deadline=time.monotonic() + 3,
                    )
                    assert result == before
                    assert request.requester is not None
                    assert result.owner != request.requester.principal_id
                else:
                    error = KeyError if outcome == "missing" else PermissionError
                    with pytest.raises(error):
                        read_storage_record(
                            right,
                            request=request,
                            expected_service_uid=os.getuid(),
                            max_bytes=8192,
                            deadline=time.monotonic() + 3,
                        )
            finally:
                right.close()
                future.result(timeout=4)
        assert right.fileno() == left.fileno() == -1
        assert ledger.record(job_id) == before
        assert ledger.transitions(job_id) == history
    finally:
        left.close()
        right.close()
        ledger.close()


@pytest.mark.parametrize(
    "fault",
    [
        "trace",
        "version",
        "extra",
        "missing-field",
        "status",
        "null-success",
        "error-record",
        "job",
        "workspace",
        "incomplete",
        "duplicate",
        "nonfinite",
        "encoding",
        "nesting",
        "disconnect",
    ],
)
def test_client_refuses_inconsistent_or_incomplete_authority_response(fault: str) -> None:
    """No malformed actual socket reply becomes a partial or unrelated job record."""
    request = decode_record_request(_request(["studio.admin"]))
    record = StudioJobRecord(
        job_id=request.job_id,
        kind="analysis",
        owner="service",
        request_id=None,
        status="pending",
        execution_model="process",
        created_at_utc="2026-09-12T00:00:00Z",
    ).to_public_dict()
    response: dict[str, object] = {
        "schema_version": "studio.storage.record.v2",
        "request_id": request.request_id,
        "status": "ok",
        "record": record,
    }
    if fault == "trace":
        response["request_id"] = "unrelated"
    elif fault == "version":
        response["schema_version"] = "studio.storage.record.v1"
    elif fault == "extra":
        response["unrecognized"] = True
    elif fault == "missing-field":
        del response["request_id"]
    elif fault == "status":
        response["status"] = "partial"
    elif fault == "null-success":
        response["record"] = None
    elif fault == "error-record":
        response["status"] = "forbidden"
    elif fault == "job":
        record["job_id"] = "sj_0000000000000002"
    elif fault == "workspace":
        record["workspace"] = "another"
    elif fault == "incomplete":
        del record["lease_owner"]
    payload = json.dumps(response).encode()
    if fault == "duplicate":
        payload = payload[:-1] + b',"status":"ok"}'
    elif fault == "nonfinite":
        payload = b'{"status":NaN}'
    elif fault == "encoding":
        payload = b"\xff"
    elif fault == "nesting":
        payload = b"[" * 2000 + b"]" * 2000
    left, right = socket.socketpair()

    def respond() -> None:
        with left:
            incoming = read_verified_frame(
                left,
                expected_uid=os.getuid(),
                max_bytes=8192,
                deadline=time.monotonic() + 3,
            )
            assert decode_record_request(incoming) == request
            if fault != "disconnect":
                write_verified_frame(
                    left,
                    payload,
                    expected_uid=os.getuid(),
                    max_bytes=8192,
                    deadline=time.monotonic() + 3,
                )

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(respond)
        try:
            with pytest.raises(EOFError if fault == "disconnect" else ValueError):
                read_storage_record(
                    right,
                    request=request,
                    expected_service_uid=os.getuid(),
                    max_bytes=8192,
                    deadline=time.monotonic() + 3,
                )
        finally:
            right.close()
            future.result(timeout=4)
    assert left.fileno() == right.fileno() == -1


@pytest.mark.parametrize("fault", ["peer", "expired", "oversize"])
def test_client_refuses_before_sending_request(fault: str) -> None:
    """Wrong service identity and invalid budgets send no request bytes."""
    left, right = socket.socketpair()
    try:
        error = {"peer": PermissionError, "expired": TimeoutError, "oversize": ValueError}[fault]
        with pytest.raises(error):
            read_storage_record(
                right,
                request=decode_record_request(_request(["studio.admin"])),
                expected_service_uid=os.getuid() + 1 if fault == "peer" else os.getuid(),
                max_bytes=1 if fault == "oversize" else 8192,
                deadline=time.monotonic() - 1 if fault == "expired" else time.monotonic() + 3,
            )
        left.settimeout(1)
        assert left.recv(1) == b""
        assert right.fileno() == -1
    finally:
        left.close()
        right.close()


def test_endpoint_reads_refuse_another_workspace_before_connecting(tmp_path: Path) -> None:
    """The configured workspace binds the request before any endpoint is reached."""
    config, ledger, _gateway = boundary(tmp_path)
    ledger.close()
    request = decode_record_request(_request(["studio.admin"])).model_copy(
        update={"workspace": "elsewhere"}
    )
    with pytest.raises(ValueError, match="configured workspace"):
        read_storage_record_at_endpoint(config, request=request)
    assert not config.socket_path.exists()


@pytest.mark.parametrize("frame", [b"", b"x" * 8193])
def test_a_listener_frame_outside_the_limit_is_refused(tmp_path: Path, frame: bytes) -> None:
    """A first frame handed over by the listener is bounded like one read here."""
    ledger = StudioJobLedger(root=tmp_path)
    left, right = socket.socketpair()
    try:
        with pytest.raises(ValueError, match="byte limit"):
            serve_record_read(
                left,
                ledger=ledger,
                gateway=PolicyGateway(InMemoryAuditSink()),
                workspace="default",
                expected_api_uid=os.getuid(),
                max_bytes=8192,
                deadline=time.monotonic() + 3,
                initial_frame=frame,
            )
    finally:
        right.close()
        ledger.close()
