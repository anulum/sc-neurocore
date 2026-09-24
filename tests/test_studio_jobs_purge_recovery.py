# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Purge failure custody

"""Storage failures during purge must not destroy a retained job's artifacts."""

from __future__ import annotations

import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from sc_neurocore.studio.platform.jobs import StudioJobContext, StudioJobManager
from sc_neurocore.studio.platform.jobs_models import StudioJobRejected


@pytest.mark.parametrize("journal_state", ["prepared", "committed"])
def test_recovery_retains_inconsistent_journal_and_artifacts(
    tmp_path: Path, journal_state: str
) -> None:
    """A dead owner does not make contradictory durable records safe to discard."""
    manager = StudioJobManager(
        root=tmp_path / "jobs",
        allowed_kinds=frozenset({"analysis"}),
        default_timeout_seconds=3.0,
    )

    def task(context: StudioJobContext) -> dict[str, object]:
        context.write_artifact("proof.txt", "retained evidence")
        return {}

    job = manager.submit(kind="analysis", owner="owner", request_id=None, task=task)
    assert manager.wait(job.job_id, 2.0).status == "completed"
    assert manager._done_events[job.job_id].wait(1.0)
    departed = subprocess.run(
        [
            sys.executable,
            "-c",
            "from sc_neurocore.studio.platform.jobs_ledger_supervisor "
            "import supervisor_identity; print(supervisor_identity())",
        ],
        capture_output=True,
        text=True,
        timeout=3.0,
        check=True,
    ).stdout.strip()
    original = manager.root / job.job_id
    identity = original.stat()
    if journal_state == "prepared":
        manager._ledger.delete(job.job_id)
    with manager._ledger.transaction() as connection:
        connection.execute(
            "INSERT INTO job_purges VALUES(?,?,?,?,?)",
            (job.job_id, departed, identity.st_dev, identity.st_ino, journal_state),
        )
    records = manager.list_records()
    journal = tuple(manager._ledger.connection().execute("SELECT * FROM job_purges").fetchone())
    manager.reconcile()
    assert manager.list_records() == records
    assert (
        tuple(manager._ledger.connection().execute("SELECT * FROM job_purges").fetchone())
        == journal
    )
    assert manager.status().pending_purge_count == 1
    assert (original / "proof.txt").read_text() == "retained evidence"


@pytest.mark.parametrize("replacement", ["directory", "missing", "file", "stage"])
def test_replaced_directory_is_preserved_after_purge_preparation(
    tmp_path: Path, replacement: str
) -> None:
    """A different directory at the original path is not covered by the purge intent."""
    manager = StudioJobManager(
        root=tmp_path,
        allowed_kinds=frozenset({"analysis"}),
        default_timeout_seconds=3.0,
    )
    job = manager.submit(kind="analysis", owner="owner", request_id=None, task=lambda ctx: {})
    assert manager.wait(job.job_id, 2.0).status == "completed"
    assert manager._done_events[job.job_id].wait(1.0)
    original_record = manager.record(job.job_id)
    work_dir, preserved = tmp_path / job.job_id, tmp_path / "preserved-original"
    staged = tmp_path / f".purge-{job.job_id}"
    statements: list[str] = []
    replaced = False

    def replace_after_prepare(statement: str) -> None:
        # Runs when the purger's connection starts its first statement after
        # the preparation committed: a real filesystem change at that point.
        nonlocal replaced
        prepared = any("'prepared'" in text for text in statements)
        if prepared and statements[-1] == "COMMIT" and not replaced:
            replaced = True
            if replacement == "stage":
                staged.mkdir()
                (staged / "foreign.txt").write_text("unrelated pending bytes")
            else:
                work_dir.rename(preserved)
            if replacement == "directory":
                work_dir.mkdir()
                (work_dir / "foreign.txt").write_text("not owned by this purge")
            elif replacement == "file":
                work_dir.write_text("unrelated replacement file")
        statements.append(statement)

    connection = manager._ledger.connection()
    connection.set_trace_callback(replace_after_prepare)
    try:
        with pytest.raises(
            StudioJobRejected, match="identity changed|disappeared|not a directory|pending purge"
        ):
            manager.purge_terminal_record(job.job_id)
    finally:
        connection.set_trace_callback(None)
    assert replaced
    assert manager.record(job.job_id) == original_record
    assert (work_dir if replacement == "stage" else preserved).is_dir()
    if replacement == "missing":
        assert not work_dir.exists()
    elif replacement == "directory":
        assert (work_dir / "foreign.txt").read_text() == "not owned by this purge"
    elif replacement == "file":
        assert work_dir.read_text() == "unrelated replacement file"
    else:
        assert (staged / "foreign.txt").read_text() == "unrelated pending bytes"
    assert (
        manager._ledger.connection().execute("SELECT COUNT(*) FROM job_purges").fetchone()[0] == 1
    )


def test_prepared_purge_blocks_direct_ledger_deletion(tmp_path: Path) -> None:
    """Another deletion surface cannot bypass an already prepared filesystem operation."""
    manager = StudioJobManager(
        root=tmp_path,
        allowed_kinds=frozenset({"analysis"}),
        default_timeout_seconds=3.0,
    )
    job = manager.submit(kind="analysis", owner="owner", request_id=None, task=lambda ctx: {})
    assert manager.wait(job.job_id, 2.0).status == "completed"
    assert manager._done_events[job.job_id].wait(1.0)
    original = manager.record(job.job_id)
    stat = (tmp_path / job.job_id).stat()
    with manager._ledger.transaction() as connection:
        connection.execute(
            "INSERT INTO job_purges VALUES(?,?,?,?, 'prepared')",
            (job.job_id, manager._ledger.supervisor, stat.st_dev, stat.st_ino),
        )
    with pytest.raises(StudioJobRejected, match="pending purge"):
        manager._ledger.delete(job.job_id)
    manager.reconcile()
    assert manager.record(job.job_id) == original
    assert (tmp_path / job.job_id).is_dir()
    assert (
        manager._ledger.connection().execute("SELECT COUNT(*) FROM job_purges").fetchone()[0] == 1
    )


@pytest.mark.parametrize("phase", ["prepared", "committed", "partial-cleanup"])
def test_restart_restores_artifacts_after_crash_during_purge(tmp_path: Path, phase: str) -> None:
    """A killed purger leaves recoverable custody, not a manifest without its bytes."""
    manager = StudioJobManager(
        root=tmp_path,
        allowed_kinds=frozenset({"analysis"}),
        default_timeout_seconds=3.0,
    )

    def task(context: StudioJobContext) -> dict[str, object]:
        context.write_artifact("proof.txt", "crash recovery evidence")
        return {}

    job = manager.submit(kind="analysis", owner="owner", request_id=None, task=task)
    assert manager.wait(job.job_id, 2.0).status == "completed"
    assert manager._done_events[job.job_id].wait(1.0)
    artifact = manager.read_artifact(job.job_id, "proof.txt")
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import os,sys
from pathlib import Path
from sc_neurocore.studio.platform.jobs import StudioJobManager
from sc_neurocore.studio.platform import jobs_purge_recovery, jobs_purge_paths
m=StudioJobManager(root=Path(sys.argv[1]),allowed_kinds=frozenset({'analysis'}),default_timeout_seconds=3)
rename=jobs_purge_paths.move_without_replace
def crash_after_rename(source,target):
    result=rename(source,target)
    assert result
    os._exit(71)
if sys.argv[3]=='prepared':
    jobs_purge_paths.move_without_replace=crash_after_rename
else:
    def crash_during_cleanup(descriptor):
        if sys.argv[3]=='partial-cleanup':
            os.unlink('proof.txt', dir_fd=descriptor)
        os._exit(71)
    jobs_purge_recovery._clear_directory=crash_during_cleanup
m.purge_terminal_record(sys.argv[2])
""",
            str(tmp_path),
            job.job_id,
            phase,
        ],
        capture_output=True,
        timeout=5.0,
    )
    assert child.returncode == 71, child.stderr
    assert (tmp_path / f".purge-{job.job_id}").is_dir()
    recovered = StudioJobManager(
        root=tmp_path,
        allowed_kinds=frozenset({"analysis"}),
        default_timeout_seconds=3.0,
    )
    if phase == "prepared":
        assert recovered.read_artifact(job.job_id, "proof.txt") == artifact
    else:
        assert recovered.list_records() == ()
    assert not (tmp_path / f".purge-{job.job_id}").exists()
    assert (
        recovered._ledger.connection().execute("SELECT COUNT(*) FROM job_purges").fetchone()[0] == 0
    )


def test_database_delete_failure_preserves_readable_artifact(tmp_path: Path) -> None:
    """A real SQLite refusal retains the record, history and exact artifact bytes."""
    manager = StudioJobManager(
        root=tmp_path,
        allowed_kinds=frozenset({"analysis"}),
        default_timeout_seconds=3.0,
    )

    def task(context: StudioJobContext) -> dict[str, object]:
        context.write_artifact("proof.txt", "irreplaceable evidence")
        return {"completed": True}

    job = manager.submit(kind="analysis", owner="owner", request_id=None, task=task)
    assert manager.wait(job.job_id, 2.0).status == "completed"
    assert manager._done_events[job.job_id].wait(1.0)
    record = manager.record(job.job_id)
    history = manager.transitions(job.job_id)
    artifact = manager.read_artifact(job.job_id, "proof.txt")
    staged = tmp_path / f".purge-{job.job_id}"
    staged.mkdir()
    custody = staged / "retained.txt"
    custody.write_text("existing recovery evidence")
    with pytest.raises(StudioJobRejected, match="pending purge"):
        manager.purge_terminal_record(job.job_id)
    assert custody.read_text() == "existing recovery evidence"
    assert manager.read_artifact(job.job_id, "proof.txt") == artifact
    custody.unlink()
    staged.rmdir()
    with manager._ledger.transaction() as connection:
        connection.execute(
            "CREATE TRIGGER refuse_purge BEFORE DELETE ON jobs "
            "BEGIN SELECT RAISE(ABORT,'injected purge storage failure'); END"
        )
    with pytest.raises(sqlite3.IntegrityError, match="injected purge storage failure"):
        manager.purge_terminal_record(job.job_id)
    assert manager.record(job.job_id) == record
    assert manager.transitions(job.job_id) == history
    assert manager.read_artifact(job.job_id, "proof.txt") == artifact
    assert not staged.exists()
    with manager._ledger.transaction() as connection:
        connection.execute("DROP TRIGGER refuse_purge")
    assert manager.purge_terminal_record(job.job_id) == record
    assert not (tmp_path / job.job_id).exists()
    assert manager.list_records() == ()
    assert not staged.exists()
