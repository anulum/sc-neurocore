# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — storage authority cancellation

"""A cancellation is recorded at the authority exactly as the embedded manager does.

Requests travel over real sockets through the service's dispatch to a real
SQLite ledger. The race is a real competing writer on its own connection,
scheduled by the SQLite trace hook when the cancel begins its transaction.
"""

from __future__ import annotations

from collections.abc import Iterator
import json
import os
from pathlib import Path
import socket
import threading
import time

import pytest
from pydantic import ValidationError

from sc_neurocore.studio.platform.jobs_ledger import StudioJobLedger
from sc_neurocore.studio.platform.jobs_ledger_supervisor import supervisor_identity
from sc_neurocore.studio.platform.jobs_models import StudioJobRecord, StudioJobStatus
from sc_neurocore.studio.platform.policy_gateway import PolicyGateway
from sc_neurocore.studio.platform.policy_models import InMemoryAuditSink
from sc_neurocore.studio.platform.storage_cancel import serve_cancel
from sc_neurocore.studio.platform.storage_cancel_client import cancel_request, exchange_cancel
from sc_neurocore.studio.platform.storage_cancel_protocol import (
    StorageCancelResponse,
    decode_cancel_request,
    decode_cancel_response,
    encode_cancel_message,
)
from sc_neurocore.studio.platform.storage_peer import read_verified_frame, write_verified_frame
from sc_neurocore.studio.platform.storage_record_protocol import StorageRequester
from tests.studio_storage_generation_support import FRAME, Authority
from tests.studio_storage_supervision_support import JOB, admit

USER = StorageRequester(principal_id="operator", roles=())


@pytest.fixture
def ledger(tmp_path: Path) -> Iterator[StudioJobLedger]:
    authority = StudioJobLedger(root=tmp_path / "authority", supervisor="storage:1:1")
    try:
        admit(authority, supervisor=supervisor_identity())
        yield authority
    finally:
        authority.close()


def _cancel(
    authority: Authority, *, workspace: str = "default", requester: StorageRequester | None = USER
) -> StudioJobRecord:
    return exchange_cancel(
        authority.connect(),
        cancel_request(workspace, JOB, requester=requester),
        expected_service_uid=os.getuid(),
        max_bytes=FRAME,
        deadline=time.monotonic() + 10,
    )


@pytest.mark.parametrize("before", ["pending", "running"])
def test_a_live_job_is_marked_cancelling_once(ledger: StudioJobLedger, before: str) -> None:
    """The first request records the transition; a repeat changes nothing."""
    if before == "running":
        ledger.transition(JOB, "running")
    authority = Authority(ledger)
    first = _cancel(authority)
    history = ledger.transitions(JOB)
    again = _cancel(authority)
    authority.join()
    assert first.status == again.status == "cancelling"
    assert history[-1]["reason"] == "cancellation requested"
    assert ledger.transitions(JOB) == history


@pytest.mark.parametrize("status", ["failed", "unknown"])
def test_a_job_the_ledger_will_not_cancel_is_returned_as_it_is(
    ledger: StudioJobLedger, status: StudioJobStatus
) -> None:
    """A stopped job is unchanged; an unknown one is refused by the ledger and returned."""
    ledger.transition(JOB, status, error="stopped" if status == "failed" else None)
    authority = Authority(ledger)
    record = _cancel(authority)
    authority.join()
    assert record.status == status


def test_a_job_ending_before_the_write_is_returned_settled(ledger: StudioJobLedger) -> None:
    """A competing writer ends the job after the read; its record is returned."""
    ran: list[bool] = []

    def compete() -> None:
        ledger.transition(JOB, "failed", error="ended elsewhere")
        ledger.close()

    def trace(statement: str) -> None:
        if statement == "BEGIN IMMEDIATE" and not ran:
            ran.append(True)
            other = threading.Thread(target=compete)
            other.start()
            other.join(timeout=10)

    authority = Authority(ledger)
    authority.before = lambda served: served.connection().set_trace_callback(trace)
    record = _cancel(authority)
    authority.join()
    assert (ran, record.status, record.error) == ([True], "failed", "ended elsewhere")


def test_policy_workspace_and_job_are_checked(ledger: StudioJobLedger) -> None:
    """Anonymous requests are denied, unknown jobs are not found, other workspaces refused."""
    authority = Authority(ledger)
    with pytest.raises(PermissionError):
        _cancel(authority, requester=None)
    with pytest.raises(KeyError):
        exchange_cancel(
            authority.connect(),
            cancel_request("default", "sj_" + "0" * 16, requester=USER),
            expected_service_uid=os.getuid(),
            max_bytes=FRAME,
            deadline=time.monotonic() + 10,
        )
    with pytest.raises(EOFError):
        _cancel(authority, workspace="elsewhere")
    authority.join(expected=(ValueError,))
    assert ledger.record(JOB).status == "pending"


def _response(**changes: object) -> bytes:
    body: dict[str, object] = {
        "schema_version": "studio.storage.cancel.v1",
        "operation": "cancel",
        "request_id": "a" * 32,
        "job_id": JOB,
        "status": "ok",
        "record": {"job_id": JOB},
    }
    body.update(changes)
    return json.dumps(body).encode()


@pytest.mark.parametrize(
    "changes,error",
    [
        ({"request_id": "b" * 32}, ValueError),
        ({"job_id": "sj_" + "0" * 16}, ValueError),
        ({"status": "ok", "record": None}, ValidationError),
        ({"status": "forbidden"}, ValidationError),
        ({"status": "forbidden", "record": None}, PermissionError),
        ({"status": "not_found", "record": None}, KeyError),
        ({"extra": 1}, ValidationError),
    ],
)
def test_replies_must_answer_the_request(
    changes: dict[str, object], error: type[Exception]
) -> None:
    """Uncorrelated, inconsistent or refused replies never become a record."""
    request = cancel_request("default", JOB, requester=USER).model_copy(
        update={"request_id": "a" * 32}
    )
    assert decode_cancel_response(_response(), request=request, max_bytes=FRAME).record
    with pytest.raises(error):
        decode_cancel_response(_response(**changes), request=request, max_bytes=FRAME)


@pytest.mark.parametrize("raw", [b"", b"\xff", b'{"a":1,"a":2}', b'{"a":NaN}', b"x" * (FRAME + 1)])
def test_malformed_requests_are_refused(raw: bytes) -> None:
    """Only an exact, bounded, unambiguous frame decodes."""
    with pytest.raises(ValueError):
        decode_cancel_request(raw, max_bytes=FRAME)
    request = cancel_request("default", JOB, requester=None)
    assert decode_cancel_request(encode_cancel_message(request), max_bytes=FRAME) == request


@pytest.mark.parametrize("workspace", ["default", ""])
def test_the_handler_reads_its_own_frame_and_refuses_no_workspace(
    ledger: StudioJobLedger, workspace: str
) -> None:
    """Called directly, the handler reads the request itself; a blank workspace refuses."""
    client, service = socket.socketpair()
    failures: list[BaseException] = []

    def serve() -> None:
        try:
            serve_cancel(
                service,
                ledger=ledger,
                gateway=PolicyGateway(InMemoryAuditSink()),
                workspace=workspace,
                expected_api_uid=os.getuid(),
                max_bytes=FRAME,
                deadline=time.monotonic() + 10,
            )
        except BaseException as exc:
            failures.append(exc)
        finally:
            ledger.close()

    thread = threading.Thread(target=serve)
    thread.start()
    request = cancel_request("default", JOB, requester=USER)
    if workspace:
        record = exchange_cancel(
            client,
            request,
            expected_service_uid=os.getuid(),
            max_bytes=FRAME,
            deadline=time.monotonic() + 10,
        )
        assert record.status == "cancelling"
    else:
        with client:
            client.settimeout(10)
            assert client.recv(1) == b""
    thread.join(timeout=10)
    assert [str(failure) for failure in failures] == (
        [] if workspace else ["storage workspace must be nonempty"]
    )


def test_a_record_of_another_job_is_refused(ledger: StudioJobLedger) -> None:
    """A correlated reply carrying another job's record never becomes the answer."""
    other = json.loads(json.dumps(ledger.record(JOB).to_public_dict()))
    other["job_id"] = "sj_" + "0" * 16
    client, service = socket.socketpair()

    def serve() -> None:
        with service:
            deadline = time.monotonic() + 10
            frame = read_verified_frame(
                service, expected_uid=os.getuid(), max_bytes=FRAME, deadline=deadline
            )
            request = decode_cancel_request(frame, max_bytes=FRAME)
            reply = StorageCancelResponse(
                schema_version="studio.storage.cancel.v1",
                operation="cancel",
                request_id=request.request_id,
                job_id=request.job_id,
                status="ok",
                record=other,
            )
            write_verified_frame(
                service,
                encode_cancel_message(reply),
                expected_uid=os.getuid(),
                max_bytes=FRAME,
                deadline=deadline,
            )

    thread = threading.Thread(target=serve)
    thread.start()
    with pytest.raises(ValueError, match="does not match"):
        exchange_cancel(
            client,
            cancel_request("default", JOB, requester=USER),
            expected_service_uid=os.getuid(),
            max_bytes=FRAME,
            deadline=time.monotonic() + 10,
        )
    thread.join(timeout=10)
