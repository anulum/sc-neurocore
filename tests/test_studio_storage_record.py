# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — storage record wire contracts

"""Verify exact read requests before authority dispatch exists."""

import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Iterator

import pytest

from sc_neurocore.studio.platform.storage_record_protocol import decode_record_request
from sc_neurocore.studio.platform.storage_record import serve_record_read
from sc_neurocore.studio.platform.jobs_ledger import StudioJobLedger
from sc_neurocore.studio.platform.jobs_snapshot import decode_job_snapshot
from sc_neurocore.studio.platform.policy_gateway import PolicyGateway
from sc_neurocore.studio.platform.policy_models import AuditEvent, AuditSinkError, InMemoryAuditSink
from sc_neurocore.studio.platform.storage_peer import read_verified_frame, write_verified_frame
from sc_neurocore.studio.training_contract import resolve_training_config


def _request() -> dict[str, object]:
    return {
        "schema_version": "studio.storage.record.v2",
        "operation": "record",
        "request_id": "trace",
        "job_id": "sj_0000000000000001",
        "workspace": "default",
        "requester": {"principal_id": "operátor", "roles": ["studio.admin"]},
    }


def test_complete_request_roundtrip_and_nullable_claim() -> None:
    """JSON reconstruction preserves version, operation, trace and requester."""
    payload = _request()
    request = decode_record_request(json.dumps(payload).encode())
    assert request.model_dump(mode="json") == payload
    assert request.requester is not None
    assert request.requester.roles == ("studio.admin",)
    payload["requester"] = None
    payload["request_id"] = None
    assert decode_record_request(json.dumps(payload).encode()).model_dump(mode="json") == payload


@pytest.mark.parametrize("field", list(_request()))
def test_every_request_field_is_required(field: str) -> None:
    """Nullable fields are explicit, not repaired with implicit defaults."""
    payload = _request()
    del payload[field]
    with pytest.raises(ValueError):
        decode_record_request(json.dumps(payload).encode())


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", "studio.storage.record.v1"),
        ("operation", "delete"),
        ("request_id", 1),
        ("job_id", ""),
        ("job_id", True),
        ("workspace", ""),
        ("workspace", []),
        ("sql", "SELECT * FROM jobs"),
        ("requester", {}),
        ("requester", {"principal_id": "", "roles": []}),
        ("requester", {"principal_id": "a", "roles": "studio.admin"}),
        ("requester", {"principal_id": "a", "roles": [1]}),
        ("requester", {"principal_id": "a", "roles": [""]}),
        ("requester", {"principal_id": "a", "roles": [], "trusted": True}),
    ],
)
def test_invalid_wire_contract_refuses(field: str, value: object) -> None:
    """Unknown operations, fields, coercions and malformed principal claims refuse."""
    payload = _request()
    payload[field] = value
    with pytest.raises(ValueError):
        decode_record_request(json.dumps(payload).encode())


@pytest.mark.parametrize(
    "payload",
    [
        b"[]",
        b"null",
        b"{}{}",
        b"\xff",
        b'{"operation":"record","operation":"delete"}',
        b'{"requester":{"roles":[],"roles":["studio.admin"]}}',
        b'{"request_id":NaN}',
        b'{"request_id":Infinity}',
        b"[" * 2000 + b"]" * 2000,
    ],
)
def test_ambiguous_encoding_and_nesting_refuse(payload: bytes) -> None:
    """Duplicate keys at any nesting level and invalid JSON never reach policy."""
    with pytest.raises(ValueError):
        decode_record_request(payload)


@pytest.fixture
def ledger(tmp_path: Path) -> Iterator[StudioJobLedger]:
    """Retain a real ledger and prove every read leaves job custody unchanged."""
    store = StudioJobLedger(root=tmp_path)
    store.create(
        job_id="sj_0000000000000001",
        kind="analysis",
        actor="studio-service",
        workspace="default",
        request_id="original",
        idempotency_key=None,
        experiment_sha256=None,
        admission=None,
        execution_model="process",
    )
    before = store.record("sj_0000000000000001")
    history = store.transitions(before.job_id)
    try:
        yield store
        assert store.record(before.job_id) == before
        assert store.transitions(before.job_id) == history
    finally:
        store.close()


def _exchange(
    ledger: StudioJobLedger, payload: dict[str, object], sink: InMemoryAuditSink
) -> tuple[bytes | None, list[Exception]]:
    left, right = socket.socketpair()
    errors: list[Exception] = []

    def serve() -> None:
        try:
            serve_record_read(
                left,
                ledger=ledger,
                gateway=PolicyGateway(sink),
                workspace="default",
                expected_api_uid=os.getuid(),
                max_bytes=8192,
                deadline=time.monotonic() + 5,
            )
        except Exception as exc:
            errors.append(exc)
        finally:
            ledger.close()

    with left, right:
        worker = threading.Thread(target=serve)
        worker.start()
        try:
            deadline = time.monotonic() + 5
            write_verified_frame(
                right,
                json.dumps(payload).encode(),
                expected_uid=os.getuid(),
                max_bytes=8192,
                deadline=deadline,
            )
            try:
                result = read_verified_frame(
                    right, expected_uid=os.getuid(), max_bytes=8192, deadline=deadline
                )
            except EOFError:
                result = None
        finally:
            worker.join(timeout=6)
            assert not worker.is_alive()
    return result, errors


@pytest.mark.parametrize("principal", [None, {"principal_id": "viewer", "roles": []}])
def test_policy_denial_returns_no_record(ledger: StudioJobLedger, principal: object) -> None:
    """Anonymous and non-admin delegation fail through the existing gateway."""
    payload = _request()
    payload["requester"] = principal
    sink = InMemoryAuditSink()
    response, errors = _exchange(ledger, payload, sink)
    assert errors == []
    assert response is not None
    assert json.loads(response) == {
        "schema_version": "studio.storage.record.v2",
        "request_id": "trace",
        "status": "forbidden",
        "record": None,
    }
    assert len(sink.events) == 1
    assert sink.events[0].decision == "deny"


def test_admin_reads_other_service_owner_without_changing_custody(ledger: StudioJobLedger) -> None:
    """Admin-global owner semantics survive real framed ledger lookup."""
    sink = InMemoryAuditSink()
    response, errors = _exchange(ledger, _request(), sink)
    assert errors == []
    assert response is not None
    envelope = json.loads(response)
    assert envelope["status"] == "ok"
    assert envelope["request_id"] == "trace"
    record = decode_job_snapshot(envelope["record"])
    assert record == ledger.record("sj_0000000000000001")
    assert record.owner == "studio-service"
    assert record.request_id == "original"
    assert sink.events[0].principal_id == "operátor"
    assert sink.events[0].request_id == "trace"
    assert sink.events[0].decision == "allow"


def test_training_record_read_preserves_config_snapshot(ledger: StudioJobLedger) -> None:
    """The real peer-verified record response retains the v7 training snapshot."""
    config = resolve_training_config({"epochs": 1, "hidden": [4]}).to_public_dict()
    job_id = "sj_0000000000000002"
    ledger.create(
        job_id=job_id,
        kind="training",
        actor="studio-training",
        workspace="default",
        request_id=None,
        idempotency_key=None,
        experiment_sha256=None,
        admission=None,
        execution_model="process",
        training_config=config,
    )
    request = _request()
    request["job_id"] = job_id

    response, errors = _exchange(ledger, request, InMemoryAuditSink())

    assert errors == []
    assert response is not None
    record = decode_job_snapshot(json.loads(response)["record"])
    assert record.job_id == job_id
    assert record.training_config == config


def test_missing_record_has_typed_outcome(ledger: StudioJobLedger) -> None:
    """Missing scoped records expose no internal exception or local path."""
    payload = _request()
    payload["job_id"] = "missing"
    response, errors = _exchange(ledger, payload, InMemoryAuditSink())
    assert errors == []
    assert response is not None
    envelope = json.loads(response)
    assert envelope["status"] == "not_found"
    assert envelope["record"] is None


def test_request_cannot_select_another_workspace(ledger: StudioJobLedger) -> None:
    """Workspace mismatch closes before policy can log a misleading allow."""
    payload = _request()
    payload["workspace"] = "another"
    sink = InMemoryAuditSink()
    response, errors = _exchange(ledger, payload, sink)
    assert response is None
    assert len(errors) == 1
    assert isinstance(errors[0], ValueError)
    assert sink.events == ()


def test_invalid_server_configuration_refuses_before_wire(ledger: StudioJobLedger) -> None:
    """Empty server workspace never waits for a peer message or accesses a job."""
    left, right = socket.socketpair()
    with left, right:
        with pytest.raises(ValueError, match="workspace"):
            serve_record_read(
                left,
                ledger=ledger,
                gateway=PolicyGateway(InMemoryAuditSink()),
                workspace="",
                expected_api_uid=os.getuid(),
                max_bytes=8192,
                deadline=time.monotonic() + 1,
            )
        assert right.recv(1) == b""


def test_exited_api_peer_cannot_reach_policy_or_ledger(
    ledger: StudioJobLedger, tmp_path: Path
) -> None:
    """The service refuses a dead connected peer before record dispatch."""
    endpoint = tmp_path / "record.sock"
    sink = InMemoryAuditSink()
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
        listener.bind(str(endpoint))
        listener.listen(1)
        child = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import socket,sys\nwith socket.socket(socket.AF_UNIX) as s: s.connect(sys.argv[1])",
                str(endpoint),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            channel, _ = listener.accept()
            _, stderr = child.communicate(timeout=5)
            assert child.returncode == 0, stderr
            with pytest.raises(PermissionError):
                serve_record_read(
                    channel,
                    ledger=ledger,
                    gateway=PolicyGateway(sink),
                    workspace="default",
                    expected_api_uid=os.getuid(),
                    max_bytes=8192,
                    deadline=time.monotonic() + 1,
                )
            assert channel.fileno() == -1
            assert sink.events == ()
        finally:
            if child.poll() is None:
                child.kill()
            child.communicate(timeout=5)


def test_audit_failure_sends_no_record(ledger: StudioJobLedger) -> None:
    """A failed policy audit cannot yield an otherwise authorized record."""

    class UnavailableSink(InMemoryAuditSink):
        def record(self, event: AuditEvent) -> None:
            raise AuditSinkError("fixture audit unavailable")

    response, errors = _exchange(ledger, _request(), UnavailableSink())
    assert response is None
    assert len(errors) == 1
    assert isinstance(errors[0], AuditSinkError)


@pytest.mark.parametrize(
    "field,value",
    [("operation", "delete"), ("schema_version", "v99"), ("sql", "SELECT * FROM jobs")],
)
def test_malformed_operation_closes_before_policy(
    ledger: StudioJobLedger, field: str, value: object
) -> None:
    """Actual framed invalid requests never reach policy or record lookup."""
    payload = _request()
    payload[field] = value
    sink = InMemoryAuditSink()
    response, errors = _exchange(ledger, payload, sink)
    assert response is None
    assert len(errors) == 1
    assert isinstance(errors[0], ValueError)
    assert sink.events == ()
