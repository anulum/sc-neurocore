# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — delegated job supervision test support

"""Real ledger admission and socket-pair exchange shared by supervision tests.

A race is exercised with a real concurrent writer: when the handler's thread
begins its second write transaction (after the ownership check, before the
acting writer), another thread with its own SQLite connection runs the
competing change and commits. The SQLite trace hook only schedules that
writer; no product code is replaced.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from datetime import UTC, datetime
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time

import pytest

from sc_neurocore.studio.platform.jobs_ledger import StudioJobLedger
from sc_neurocore.studio.platform.jobs_ledger_supervisor import supervisor_identity
from sc_neurocore.studio.platform.jobs_shared_admission import SharedJobAdmission
from sc_neurocore.studio.platform.storage_supervision import serve_supervision
from sc_neurocore.studio.platform.storage_supervision_client import exchange_supervision
from sc_neurocore.studio.platform.storage_supervision_protocol import (
    SUPERVISION_SCHEMA_VERSION,
    StorageSupervisionResponse,
    SupervisionHeartbeatRequest,
    SupervisionStartRequest,
)

JOB = "sj_" + "7" * 16


class Clock:
    """Controllable UTC clock so lease renewal is observable at second resolution."""

    def __init__(self) -> None:
        self.now = datetime(2026, 9, 24, 0, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now


def admit(ledger: StudioJobLedger, *, supervisor: str, workspace: str = "default") -> None:
    """Admit one process job with its lease and reservation delegated to ``supervisor``."""
    SharedJobAdmission(ledger, max_concurrent=2, max_queued=0).admit(
        job_id=JOB,
        kind="analysis",
        actor="operator",
        workspace=workspace,
        request_id=None,
        idempotency_key=None,
        experiment_sha256=None,
        admission=None,
        execution_model="process",
        supervisor=supervisor,
    )


def start(worker: str, *, workspace: str = "default") -> SupervisionStartRequest:
    """Build a start request with a fixed request ID."""
    return SupervisionStartRequest(
        schema_version=SUPERVISION_SCHEMA_VERSION,
        operation="start",
        request_id="a" * 32,
        workspace=workspace,
        job_id=JOB,
        worker=worker,
    )


def heartbeat() -> SupervisionHeartbeatRequest:
    """Build a heartbeat request with a fixed request ID."""
    return SupervisionHeartbeatRequest(
        schema_version=SUPERVISION_SCHEMA_VERSION,
        operation="heartbeat",
        request_id="b" * 32,
        workspace="default",
        job_id=JOB,
    )


def _run_before_acting_write(
    ledger: StudioJobLedger,
    competing: Callable[[], None],
    ran: list[bool],
    failures: list[BaseException],
) -> None:
    """Run ``competing`` from another thread before this thread's second write."""
    begun = 0

    def concurrent() -> None:
        try:
            competing()
            ran.append(True)
        except BaseException as exc:
            failures.append(exc)
        finally:
            ledger.close()

    def trace(statement: str) -> None:
        nonlocal begun
        if statement == "BEGIN IMMEDIATE":
            begun += 1
            if begun == 2:
                thread = threading.Thread(target=concurrent)
                thread.start()
                thread.join(timeout=10.0)

    ledger.connection().set_trace_callback(trace)


def exchange(
    ledger: StudioJobLedger,
    request: SupervisionStartRequest | SupervisionHeartbeatRequest,
    *,
    competing: Callable[[], None] | None = None,
) -> StorageSupervisionResponse:
    """Run the real service handler and the real client over one socket pair.

    ``competing`` runs as a concurrent writer before the handler acts; the
    exchange asserts that it ran and committed.
    """
    service, client = socket.socketpair()
    failures: list[BaseException] = []
    ran: list[bool] = []

    def serve() -> None:
        try:
            if competing is not None:
                _run_before_acting_write(ledger, competing, ran, failures)
            serve_supervision(
                service,
                ledger=ledger,
                workspace="default",
                expected_api_uid=os.getuid(),
                max_bytes=4096,
                deadline=time.monotonic() + 10.0,
            )
        except BaseException as exc:
            failures.append(exc)
        finally:
            ledger.close()

    thread = threading.Thread(target=serve)
    thread.start()
    try:
        return exchange_supervision(
            client,
            request,
            expected_service_uid=os.getuid(),
            max_bytes=4096,
            deadline=time.monotonic() + 10.0,
        )
    finally:
        thread.join(timeout=10.0)
        assert not thread.is_alive()
        assert failures == []
        assert ran == ([] if competing is None else [True])


def worker_rows(ledger: StudioJobLedger) -> list[tuple[str, str]]:
    rows = ledger.connection().execute("SELECT supervisor, worker_identity FROM job_workers")
    return [(str(row[0]), str(row[1])) for row in rows.fetchall()]


@pytest.fixture
def clock() -> Clock:
    """Clock shared by the ledger fixture and the test."""
    return Clock()


@pytest.fixture
def ledger(tmp_path: Path, clock: Clock) -> Iterator[StudioJobLedger]:
    """Storage authority ledger whose own supervisor is the storage service."""
    authority = StudioJobLedger(root=tmp_path / "authority", supervisor="storage:1:1", clock=clock)
    try:
        yield authority
    finally:
        authority.close()


@pytest.fixture
def worker() -> Iterator[str]:
    """A live process leading its own group, as a launched worker does."""
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True
    )
    try:
        yield supervisor_identity(child.pid)
    finally:
        child.kill()
        child.wait(timeout=10.0)
