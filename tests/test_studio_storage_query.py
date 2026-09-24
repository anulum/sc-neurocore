# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — storage authority read-only views

"""The API reads records, status and the purge journal through the authority.

Every exchange runs over real sockets through the service's own dispatch, over
a real SQLite ledger whose jobs were admitted by the real shared admission.
Pages are cut by the real frame limit; policy is the existing gateway.
"""

from __future__ import annotations

from collections.abc import Iterator
import os
from pathlib import Path
import socket
import threading
import time

import pytest

from sc_neurocore.studio.platform.jobs_ledger import StudioJobLedger
from sc_neurocore.studio.platform.jobs_ledger_supervisor import supervisor_identity
from sc_neurocore.studio.platform.jobs_models import StudioJobExecutionModel
from sc_neurocore.studio.platform.jobs_shared_admission import SharedJobAdmission
from sc_neurocore.studio.platform.policy_gateway import PolicyGateway
from sc_neurocore.studio.platform.policy_models import InMemoryAuditSink
from sc_neurocore.studio.platform.storage_query import serve_query
from sc_neurocore.studio.platform.storage_query_client import (
    QueryReader,
    exchange_query,
    query_request,
)
from sc_neurocore.studio.platform.storage_record_protocol import StorageRequester
from tests.studio_storage_generation_support import FRAME, Authority
from tests.studio_storage_supervision_support import Clock

ADMIN = StorageRequester(principal_id="operator", roles=("studio.admin",))
VIEWER = StorageRequester(principal_id="viewer", roles=("studio.viewer",))


@pytest.fixture
def ledger(tmp_path: Path) -> Iterator[StudioJobLedger]:
    """Authority ledger with a frozen clock, so creation times tie."""
    authority = StudioJobLedger(
        root=tmp_path / "authority", supervisor="storage:1:1", clock=Clock()
    )
    try:
        yield authority
    finally:
        authority.close()


def _admit(
    ledger: StudioJobLedger,
    count: int,
    *,
    workspace: str = "default",
    model: StudioJobExecutionModel = "process",
    first: int = 0,
) -> list[str]:
    """Admit ``count`` jobs whose IDs sort against their creation order."""
    admission = SharedJobAdmission(ledger, max_concurrent=64, max_queued=0)
    ids = [f"sj_{0xFFFF - first - index:016x}" for index in range(count)]
    for job_id in ids:
        admission.admit(
            job_id=job_id,
            kind="analysis",
            actor="operator",
            workspace=workspace,
            request_id=None,
            idempotency_key=None,
            experiment_sha256=None,
            admission=None,
            execution_model=model,
            supervisor=supervisor_identity(),
        )
    return ids


def _reader(authority: Authority, max_bytes: int = FRAME) -> QueryReader:
    return QueryReader(
        authority.connect,
        workspace="default",
        storage_uid=os.getuid(),
        max_bytes=max_bytes,
        timeout_seconds=10.0,
    )


def test_record_pages_follow_creation_order_within_the_workspace(
    ledger: StudioJobLedger,
) -> None:
    """Pages cut by the frame still return every record, in creation order."""
    ids = _admit(ledger, 5)
    _admit(ledger, 2, workspace="elsewhere", first=10)
    one = len(ledger.record(ids[0]).to_public_dict().__repr__())
    authority = Authority(ledger, frame_max_bytes=3 * one)
    records = _reader(authority, 3 * one).records(ADMIN)
    authority.join()
    assert [record.job_id for record in records] == ids
    assert records == ledger.list_records(workspace="default")
    assert authority.seen.count("query") >= 3


def test_a_record_larger_than_the_frame_is_a_configuration_fault(
    ledger: StudioJobLedger,
) -> None:
    """The authority refuses rather than truncating a record; the API sees the close."""
    _admit(ledger, 1)
    authority = Authority(ledger, frame_max_bytes=512)
    with pytest.raises(EOFError):
        _reader(authority, 512).records(ADMIN)
    authority.join(expected=(ValueError,))


def test_unknown_cursors_and_foreign_workspaces_are_refused(ledger: StudioJobLedger) -> None:
    """A cursor must name a job of the workspace; another workspace is never served."""
    _admit(ledger, 1, workspace="elsewhere")
    authority = Authority(ledger)
    request = query_request("default", "records", requester=ADMIN, after="sj_" + "0" * 16)
    with pytest.raises(ValueError, match="cursor"):
        exchange_query(
            authority.connect(),
            request,
            expected_service_uid=os.getuid(),
            max_bytes=FRAME,
            deadline=time.monotonic() + 10,
        )
    foreign = query_request("elsewhere", "records", requester=ADMIN)
    with pytest.raises(EOFError):
        exchange_query(
            authority.connect(),
            foreign,
            expected_service_uid=os.getuid(),
            max_bytes=FRAME,
            deadline=time.monotonic() + 10,
        )
    authority.join(expected=(ValueError,))


def test_each_view_applies_the_policy_of_its_route(ledger: StudioJobLedger) -> None:
    """Records and purges need the admin role; status is public, as over HTTP."""
    _admit(ledger, 1)
    authority = Authority(ledger)
    reader = _reader(authority)
    for requester in (None, VIEWER):
        with pytest.raises(PermissionError):
            reader.records(requester)
        with pytest.raises(PermissionError):
            reader.purges(requester, limit=10, after=None)
    assert reader.status(None).statuses == {"pending": 1}
    authority.join()
    denied = {event.route for event in authority.audit.events if event.decision != "allow"}
    assert denied == {"/api/studio/jobs", "/api/studio/jobs/purges"}


def test_status_counts_the_workspace_and_its_admission(ledger: StudioJobLedger) -> None:
    """Counts by status and model, pending purges and admission occupancy."""
    ids = _admit(ledger, 3)
    _admit(ledger, 1, model="thread", first=5)
    _admit(ledger, 1, workspace="elsewhere", first=9)
    ledger.transition(ids[0], "running")
    ledger.transition(ids[1], "failed", error="stopped")
    with ledger.transaction() as connection:
        connection.execute(
            "INSERT INTO job_purges VALUES(?,?,?,?,?)",
            ("sj_" + "a" * 16, "supervisor", None, None, "committed"),
        )
    authority = Authority(ledger)
    summary = _reader(authority).status(ADMIN)
    authority.join()
    assert summary.statuses == {"failed": 1, "pending": 2, "running": 1}
    assert summary.execution_models == {"process": 3, "thread": 1}
    assert (summary.pending_purge_count, summary.unreaped) == (1, [])
    assert summary.admission["max_concurrent"] == 64


def test_the_purge_journal_is_paged_by_job_id(ledger: StudioJobLedger) -> None:
    """Journal entries arrive in lexical pages with their recorded identity."""
    rows = [(f"sj_{index:016x}", "supervisor", 7, index, "prepared") for index in (3, 1, 2)]
    with ledger.transaction() as connection:
        connection.executemany("INSERT INTO job_purges VALUES(?,?,?,?,?)", rows)
    authority = Authority(ledger)
    reader = _reader(authority)
    first = reader.purges(ADMIN, limit=2, after=None)
    second = reader.purges(ADMIN, limit=2, after=first.next_after)
    authority.join()
    assert [purge.job_id for purge in first.purges] == [
        "sj_" + "0" * 15 + "1",
        "sj_" + "0" * 15 + "2",
    ]
    assert first.next_after == "sj_" + "0" * 15 + "2"
    assert [(purge.job_id, purge.inode) for purge in second.purges] == [("sj_" + "0" * 15 + "3", 3)]
    assert second.next_after is None


def test_the_handler_reads_its_own_first_frame_and_refuses_no_workspace(
    ledger: StudioJobLedger,
) -> None:
    """Called directly, the handler reads the request itself; a blank workspace refuses."""
    _admit(ledger, 1)
    admission = SharedJobAdmission(ledger, max_concurrent=64, max_queued=0)
    for workspace in ("default", ""):
        client, service = socket.socketpair()
        failures: list[BaseException] = []

        def serve(target: str = workspace, channel: socket.socket = service) -> None:
            try:
                serve_query(
                    channel,
                    ledger=ledger,
                    admission=admission,
                    gateway=PolicyGateway(InMemoryAuditSink()),
                    workspace=target,
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
            response = exchange_query(
                client,
                query_request("default", "status", requester=None),
                expected_service_uid=os.getuid(),
                max_bytes=FRAME,
                deadline=time.monotonic() + 10,
            )
            assert response.summary is not None
        else:
            with client:
                client.settimeout(10)
                assert client.recv(1) == b""
        thread.join(timeout=10)
        assert [str(failure) for failure in failures] == (
            [] if workspace else ["storage workspace must be nonempty"]
        )
