# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — storage authority purge

"""The authority purges a terminal job and its sealed directory, or says why not.

Jobs are sealed by the real finish exchange; purges go over real sockets
through the service's dispatch and run the embedded durable purge phases on
the authority's own root.
"""

from __future__ import annotations

import json
import os
import socket
import threading
import time
from pathlib import Path

import pytest
from pydantic import ValidationError

from sc_neurocore.studio.platform.jobs_ledger import StudioJobLedger
from sc_neurocore.studio.platform.jobs_ledger_supervisor import supervisor_identity
from sc_neurocore.studio.platform.jobs_models import StudioJobRecord, StudioJobRejected
from sc_neurocore.studio.platform.policy_gateway import PolicyGateway
from sc_neurocore.studio.platform.policy_models import InMemoryAuditSink
from sc_neurocore.studio.platform.storage_peer import read_verified_frame, write_verified_frame
from sc_neurocore.studio.platform.storage_purge import AuthorityCustody, serve_purge
from sc_neurocore.studio.platform.storage_purge_client import exchange_purge, purge_request
from sc_neurocore.studio.platform.storage_purge_protocol import (
    StoragePurgeResponse,
    decode_purge_request,
    decode_purge_response,
    encode_purge_message,
)
from sc_neurocore.studio.platform.storage_record_protocol import StorageRequester
from tests.studio_storage_finish_support import FILES, finish, request, started, stop
from tests.studio_storage_generation_support import FRAME, Authority
from tests.studio_storage_supervision_support import *

ADMIN = StorageRequester(principal_id="operator", roles=("studio.admin",))


def _purge(
    authority: Authority,
    *,
    requester: StorageRequester | None = ADMIN,
    job_id: str = JOB,
    workspace: str = "default",
) -> StudioJobRecord:
    return exchange_purge(
        authority.connect(),
        purge_request(workspace, job_id, requester=requester),
        expected_service_uid=os.getuid(),
        max_bytes=FRAME,
        deadline=time.monotonic() + 10,
    )


def test_a_sealed_terminal_job_is_purged_with_its_directory(ledger: StudioJobLedger) -> None:
    """The record, its sealed bytes and the purge journal entry are all gone."""
    stop(started(ledger))
    assert finish(ledger, request(FILES), list(FILES.values())).reply == "sealed"
    sealed = ledger.path.parent / JOB
    assert sealed.is_dir()
    authority = Authority(ledger)
    purged = _purge(authority)
    authority.join()
    assert (purged.job_id, purged.status) == (JOB, "completed")
    with pytest.raises(KeyError):
        ledger.record(JOB)
    assert not sealed.exists() and not (ledger.path.parent / f".purge-{JOB}").exists()
    assert ledger.connection().execute("SELECT COUNT(*) FROM job_purges").fetchone()[0] == 0


@pytest.mark.parametrize("state", ["running", "unreaped"])
def test_live_or_unreaped_jobs_are_refused_with_the_ledger_reason(
    ledger: StudioJobLedger, state: str
) -> None:
    """An active job, or a terminal one still holding capacity, is not purged."""
    worker = started(ledger)
    stop(worker)
    if state == "unreaped":
        assert (
            finish(ledger, request({}, outcome="cancelled", worker_reaped=False), []).reply
            == "sealed"
        )
    authority = Authority(ledger)
    with pytest.raises(StudioJobRejected) as refused:
        _purge(authority)
    authority.join()
    assert str(refused.value)
    assert ledger.record(JOB).job_id == JOB


def test_policy_workspace_and_job_are_checked(ledger: StudioJobLedger) -> None:
    """Non-admins are denied, unknown jobs not found, another workspace refused."""
    admit(ledger, supervisor=supervisor_identity())
    authority = Authority(ledger)
    with pytest.raises(PermissionError):
        _purge(authority, requester=StorageRequester(principal_id="viewer", roles=()))
    with pytest.raises(KeyError):
        _purge(authority, job_id="sj_" + "0" * 16)
    with pytest.raises(EOFError):
        _purge(authority, workspace="elsewhere")
    authority.join(expected=(ValueError,))


def _reply(request_id: str, changes: dict[str, object] | None = None) -> bytes:
    body: dict[str, object] = {
        "schema_version": "studio.storage.purge.v1",
        "operation": "purge",
        "request_id": request_id,
        "job_id": JOB,
        "status": "ok",
        "record": {"job_id": JOB},
        "error": None,
    }
    body.update(changes or {})
    return json.dumps(body).encode()


@pytest.mark.parametrize(
    "changes,error",
    [
        ({"request_id": "0" * 32}, ValueError),
        ({"record": None}, ValidationError),
        ({"error": "x"}, ValidationError),
        ({"status": "refused", "record": None}, ValidationError),
        ({"status": "forbidden", "record": None}, PermissionError),
        ({"status": "not_found", "record": None}, KeyError),
        ({"status": "refused", "record": None, "error": "reserved"}, StudioJobRejected),
    ],
)
def test_replies_must_answer_the_request(
    changes: dict[str, object], error: type[Exception]
) -> None:
    """Uncorrelated or inconsistent replies never decode; refusals raise their error."""
    sent = purge_request("default", JOB, requester=ADMIN)
    assert decode_purge_response(_reply(sent.request_id), request=sent, max_bytes=FRAME).record
    with pytest.raises(error):
        decode_purge_response(_reply(sent.request_id, changes), request=sent, max_bytes=FRAME)
    for raw in (b"", b"\xff", b'{"a":1,"a":2}', b'{"a":NaN}'):
        with pytest.raises(ValueError):
            decode_purge_request(raw, max_bytes=FRAME)
    assert decode_purge_request(encode_purge_message(sent), max_bytes=FRAME) == sent


@pytest.mark.parametrize("workspace", ["default", ""])
def test_the_handler_reads_its_own_frame_and_refuses_no_workspace(
    ledger: StudioJobLedger, workspace: str
) -> None:
    """Called directly, the handler reads the request itself; a blank workspace refuses."""
    client, service = socket.socketpair()
    failures: list[BaseException] = []

    def serve() -> None:
        try:
            serve_purge(
                service,
                custody=AuthorityCustody(ledger),
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
    if workspace:
        with pytest.raises(KeyError):
            exchange_purge(
                client,
                purge_request("default", JOB, requester=ADMIN),
                expected_service_uid=os.getuid(),
                max_bytes=FRAME,
                deadline=time.monotonic() + 10,
            )
    else:
        with client:
            client.settimeout(10)
            assert client.recv(1) == b""
    thread.join(timeout=10)
    assert [str(failure) for failure in failures] == (
        [] if workspace else ["storage workspace must be nonempty"]
    )


def test_a_record_of_another_job_is_refused(ledger: StudioJobLedger, tmp_path: Path) -> None:
    """A correlated purge reply carrying another job's record never becomes the answer."""
    admit(ledger, supervisor=supervisor_identity())
    other = json.loads(json.dumps(ledger.record(JOB).to_public_dict()))
    other["job_id"] = "sj_" + "0" * 16
    client, service = socket.socketpair()

    def serve() -> None:
        with service:
            deadline = time.monotonic() + 10
            frame = read_verified_frame(
                service, expected_uid=os.getuid(), max_bytes=FRAME, deadline=deadline
            )
            sent = decode_purge_request(frame, max_bytes=FRAME)
            reply = StoragePurgeResponse(
                schema_version="studio.storage.purge.v1",
                operation="purge",
                request_id=sent.request_id,
                job_id=sent.job_id,
                status="ok",
                record=other,
                error=None,
            )
            write_verified_frame(
                service,
                encode_purge_message(reply),
                expected_uid=os.getuid(),
                max_bytes=FRAME,
                deadline=deadline,
            )

    thread = threading.Thread(target=serve)
    thread.start()
    with pytest.raises(ValueError, match="does not match"):
        exchange_purge(
            client,
            purge_request("default", JOB, requester=ADMIN),
            expected_service_uid=os.getuid(),
            max_bytes=FRAME,
            deadline=time.monotonic() + 10,
        )
    thread.join(timeout=10)
