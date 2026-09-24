# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Ledger commit failure recovery

"""Real SQLite commit refusals must release write locks and retain durable custody."""

import sqlite3
from pathlib import Path

import pytest

from sc_neurocore.studio.platform.jobs import StudioJobContext, StudioJobManager
from sc_neurocore.studio.platform.jobs_ledger import StudioJobLedger


def _deferred_constraint(connection: sqlite3.Connection) -> None:
    """Install a test-only constraint SQLite validates at commit, not insertion."""
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("CREATE TABLE commit_parent(id INTEGER PRIMARY KEY)")
    connection.execute(
        "CREATE TABLE commit_child(parent_id INTEGER REFERENCES commit_parent(id) "
        "DEFERRABLE INITIALLY DEFERRED)"
    )


def test_commit_refusal_rolls_back_and_releases_writer(tmp_path: Path) -> None:
    """A failed COMMIT releases the lock, rolls back writes, and permits a valid retry."""
    ledger = StudioJobLedger(root=tmp_path)
    connection = ledger.connection()
    _deferred_constraint(connection)
    with (
        pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY constraint failed"),
        ledger.transaction() as transaction,
    ):
        transaction.execute("INSERT INTO commit_child VALUES(1)")
        # The body succeeds; SQLite itself refuses only the later COMMIT.
        assert transaction.execute("SELECT COUNT(*) FROM commit_child").fetchone()[0] == 1
    assert not connection.in_transaction
    assert connection.execute("SELECT COUNT(*) FROM commit_child").fetchone()[0] == 0
    with sqlite3.connect(ledger.path, timeout=0.2) as observer:
        observer.execute("BEGIN IMMEDIATE")
        assert observer.execute("SELECT COUNT(*) FROM commit_child").fetchone()[0] == 0
        observer.rollback()
    with ledger.transaction() as transaction:
        transaction.execute("INSERT INTO commit_parent VALUES(1)")
        transaction.execute("INSERT INTO commit_child VALUES(1)")
    with sqlite3.connect(ledger.path, timeout=0.2) as observer:
        assert observer.execute("SELECT parent_id FROM commit_child").fetchall() == [(1,)]
    ledger.close()


def test_purge_cleanup_commit_refusal_retains_journal_without_writer_lock(tmp_path: Path) -> None:
    """Failure after filesystem cleanup preserves durable intent, not a stuck transaction."""
    manager = StudioJobManager(
        root=tmp_path, allowed_kinds=frozenset({"analysis"}), default_timeout_seconds=3.0
    )
    job = manager.submit(kind="analysis", owner="owner", request_id=None, task=lambda ctx: {})
    assert manager.wait(job.job_id, 2.0).status == "completed"
    assert manager._done_events[job.job_id].wait(1.0)
    inode = (manager.root / job.job_id).stat().st_ino
    connection = manager._ledger.connection()
    _deferred_constraint(connection)
    connection.execute(
        "CREATE TRIGGER defer_purge_refusal AFTER DELETE ON job_purges "
        "BEGIN INSERT INTO commit_child VALUES(1); END"
    )
    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY constraint failed"):
        manager.purge_terminal_record(job.job_id)
    assert not connection.in_transaction
    with sqlite3.connect(manager.ledger_path, timeout=0.2) as observer:
        observer.execute("BEGIN IMMEDIATE")
        assert observer.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0
        assert observer.execute("SELECT state,inode FROM job_purges").fetchall() == [
            ("removed", inode)
        ]
        assert observer.execute("SELECT COUNT(*) FROM commit_child").fetchone()[0] == 0
        observer.rollback()
    connection.execute("DROP TRIGGER defer_purge_refusal")
    assert not (manager.root / job.job_id).exists()
    assert not (manager.root / f".purge-{job.job_id}").exists()
    for _ in range(2):
        manager.reconcile()
        assert manager.status().pending_purge_count == 0
    reopened = StudioJobManager(
        root=tmp_path, allowed_kinds=frozenset({"analysis"}), default_timeout_seconds=3.0
    )
    assert reopened.status().pending_purge_count == 0
    assert reopened.list_records() == ()


@pytest.mark.parametrize("refused_phase", ["cleanup_started", "removed"])
def test_phase_commit_refusal_preserves_the_actual_cleanup_boundary(
    tmp_path: Path, refused_phase: str
) -> None:
    """SQLite itself refuses a phase commit; filesystem effects must match durable evidence."""
    manager = StudioJobManager(
        root=tmp_path, allowed_kinds=frozenset({"analysis"}), default_timeout_seconds=3.0
    )

    def task(context: StudioJobContext) -> dict[str, object]:
        context.write_artifact("proof.txt", "preserved before cleanup start")
        return {}

    job = manager.submit(kind="analysis", owner="owner", request_id=None, task=task)
    assert manager.wait(job.job_id, 2.0).status == "completed"
    assert manager._done_events[job.job_id].wait(1.0)
    inode = (manager.root / job.job_id).stat().st_ino
    connection = manager._ledger.connection()
    _deferred_constraint(connection)
    connection.execute("CREATE TABLE refused_phase(name TEXT NOT NULL)")
    connection.execute("INSERT INTO refused_phase VALUES(?)", (refused_phase,))
    connection.execute(
        "CREATE TRIGGER refuse_phase_commit AFTER UPDATE OF state ON job_purges "
        "WHEN NEW.state=(SELECT name FROM refused_phase) "
        "BEGIN INSERT INTO commit_child VALUES(1); END"
    )
    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY constraint failed"):
        manager.purge_terminal_record(job.job_id)
    assert not connection.in_transaction
    prior_phase = "committed" if refused_phase == "cleanup_started" else "cleanup_started"
    with sqlite3.connect(manager.ledger_path, timeout=0.2) as observer:
        observer.execute("BEGIN IMMEDIATE")
        assert observer.execute("SELECT state,inode FROM job_purges").fetchall() == [
            (prior_phase, inode)
        ]
        assert observer.execute("SELECT COUNT(*) FROM commit_child").fetchone()[0] == 0
        assert observer.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0
        observer.rollback()
    stage = manager.root / f".purge-{job.job_id}"
    if refused_phase == "cleanup_started":
        assert stage.stat().st_ino == inode
        assert (stage / "proof.txt").read_text() == "preserved before cleanup start"
    else:
        assert not stage.exists()
    connection.execute("DROP TRIGGER refuse_phase_commit")
    for _ in range(2):
        manager.reconcile()
        assert manager.status().pending_purge_count == (refused_phase == "removed")
    reopened = StudioJobManager(
        root=tmp_path, allowed_kinds=frozenset({"analysis"}), default_timeout_seconds=3.0
    )
    assert reopened.list_records() == ()
    assert reopened.status().pending_purge_count == (refused_phase == "removed")
    assert not stage.exists()
    if refused_phase == "removed":
        assert reopened._ledger.connection().execute(
            "SELECT state,inode FROM job_purges"
        ).fetchall()[0][:] == ("ambiguous", inode)
