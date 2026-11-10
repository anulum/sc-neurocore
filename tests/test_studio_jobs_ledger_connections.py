# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio job ledger connection lifecycle

"""Ledger connections close without crashing a live thread, and none outlives its ledger."""

from __future__ import annotations

import gc
import sqlite3
import threading
from pathlib import Path

from starlette.testclient import TestClient

from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.platform import StudioRuntimeSettings
from sc_neurocore.studio.platform.jobs import StudioJobManager
from sc_neurocore.studio.platform.jobs_ledger import StudioJobLedger
from sc_neurocore.studio.platform.storage_isolated_jobs import IsolatedJobManager


def _closed(connection: sqlite3.Connection) -> bool:
    try:
        connection.execute("SELECT 1")
    except sqlite3.ProgrammingError:
        return True
    return False


def _open_in_threads(ledger: StudioJobLedger, count: int) -> list[sqlite3.Connection]:
    """Open one ledger connection on each of ``count`` threads that then finish.

    The threads wait for each other before finishing, so no thread has ended
    while another opens its connection (which would close the ended one's).
    """
    opened: list[sqlite3.Connection] = []
    all_open = threading.Barrier(count)

    def use() -> None:
        connection = ledger.connection()
        connection.execute("SELECT COUNT(*) FROM jobs").fetchone()
        opened.append(connection)
        all_open.wait(timeout=30)

    threads = [threading.Thread(target=use) for _ in range(count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return opened


def test_close_all_closes_the_callers_and_every_ended_threads_connection(tmp_path: Path) -> None:
    """Connections of threads that have ended close too, not only the caller's."""
    ledger = StudioJobLedger(root=tmp_path)
    connections = [ledger.connection(), *_open_in_threads(ledger, 3)]
    assert not any(_closed(connection) for connection in connections)

    ledger.close_all()

    assert all(_closed(connection) for connection in connections)
    # A later use on any thread opens a fresh connection instead of the closed one.
    assert ledger.connection().execute("SELECT 1").fetchone()[0] == 1
    assert not any(_closed(connection) for connection in _open_in_threads(ledger, 1))
    ledger.close_all()


def test_a_finished_threads_connection_closes_when_another_thread_opens_one(
    tmp_path: Path,
) -> None:
    """A thread that ended cannot close its connection; the next opener does."""
    ledger = StudioJobLedger(root=tmp_path)
    (finished,) = _open_in_threads(ledger, 1)
    assert not _closed(finished)

    _open_in_threads(ledger, 1)

    assert _closed(finished)
    ledger.close_all()


def test_close_all_retires_a_live_threads_connection_for_its_owner_to_close(
    tmp_path: Path,
) -> None:
    """A live thread may be inside a transaction, so its connection is left to it.

    Closing it from another thread would pull the connection out from under a
    running statement (an interpreter crash, not an exception).
    """
    ledger = StudioJobLedger(root=tmp_path)
    opened, reopened = threading.Event(), threading.Event()
    release = threading.Event()
    connections: list[sqlite3.Connection] = []

    def hold_then_reuse() -> None:
        connections.append(ledger.connection())
        opened.set()
        release.wait(timeout=30)
        connections.append(ledger.connection())
        reopened.set()

    thread = threading.Thread(target=hold_then_reuse)
    thread.start()
    try:
        assert opened.wait(timeout=30)
        ledger.close_all()
        assert not _closed(connections[0])
        release.set()
        assert reopened.wait(timeout=30)
    finally:
        release.set()
        thread.join()
    assert connections[1] is not connections[0]
    assert _closed(connections[0])
    ledger.close_all()


def test_close_still_closes_only_the_callers_connection(tmp_path: Path) -> None:
    """The per-thread ``close`` keeps its contract; other threads stay open."""
    ledger = StudioJobLedger(root=tmp_path)
    mine = ledger.connection()
    started, release = threading.Event(), threading.Event()
    other: list[sqlite3.Connection] = []

    def hold() -> None:
        other.append(ledger.connection())
        started.set()
        release.wait(timeout=30)

    thread = threading.Thread(target=hold)
    thread.start()
    started.wait(timeout=30)
    ledger.close()
    try:
        assert _closed(mine)
        assert not _closed(other[0])
    finally:
        release.set()
        thread.join()
        ledger.close_all()


def test_a_collected_ledger_closes_every_connection_it_opened(tmp_path: Path) -> None:
    """Dropping a ledger without closing it still closes all of its connections."""
    ledger = StudioJobLedger(root=tmp_path)
    connections = [ledger.connection(), *_open_in_threads(ledger, 2)]

    del ledger
    gc.collect()

    assert all(_closed(connection) for connection in connections)


def test_closing_a_manager_closes_its_ledger(tmp_path: Path) -> None:
    """``StudioJobManager.close`` releases the ledger connections of every thread."""
    manager = StudioJobManager(
        root=tmp_path, allowed_kinds=frozenset({"analysis"}), default_timeout_seconds=30.0
    )
    connections = [manager._ledger.connection(), *_open_in_threads(manager._ledger, 2)]

    manager.close()
    manager.close()

    assert all(_closed(connection) for connection in connections)


def test_application_shutdown_closes_the_job_ledger(tmp_path: Path) -> None:
    """Leaving the application's lifespan closes what no live thread holds, retires the rest."""
    app = create_app(StudioRuntimeSettings(job_root_path=str(tmp_path / "jobs")))
    ledger = app.state.studio_job_manager._ledger
    with TestClient(app, base_url="http://127.0.0.1", headers={"sec-fetch-site": "none"}):
        mine = ledger.connection()
        ended = _open_in_threads(ledger, 2)
        assert not any(_closed(connection) for connection in [mine, *ended])

    assert all(_closed(connection) for connection in ended)
    # Shutdown ran on another thread; this live thread's connection was retired, not
    # closed under it, and is closed by this thread's next use of the ledger.
    assert not _closed(mine)
    assert ledger.connection() is not mine
    assert _closed(mine)
    ledger.close_all()


def test_application_shutdown_passes_by_a_manager_that_owns_no_ledger(tmp_path: Path) -> None:
    """The isolated job manager owns no ledger and has no ``close``; shutdown leaves it alone.

    In isolated mode every durable job operation goes to the storage authority,
    so the application's shutdown step finds nothing to close and must not fail.
    """
    assert not hasattr(IsolatedJobManager, "close")
    app = create_app(StudioRuntimeSettings(job_root_path=str(tmp_path / "jobs")))
    local = app.state.studio_job_manager
    app.state.studio_job_manager = object.__new__(IsolatedJobManager)
    try:
        with TestClient(app, base_url="http://127.0.0.1", headers={"sec-fetch-site": "none"}):
            pass
    finally:
        local.close()
