# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Purge filesystem commit ordering

"""Directory changes must be synchronized before dependent database commits."""

import errno
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from sc_neurocore.studio.platform.jobs import StudioJobManager
from tests.studio_purge_child_support import PURGE_PROLOGUE, reopened
from tests.studio_seccomp_support import SECCOMP_AVAILABLE, run_child

_held_calls = pytest.mark.skipif(
    not SECCOMP_AVAILABLE, reason="held system calls need Linux x86_64"
)


@_held_calls
@pytest.mark.parametrize("failed_sync", [1, 2])
def test_sync_failure_preserves_the_correct_recovery_boundary(
    tmp_path: Path, failed_sync: int
) -> None:
    """Pre-delete sync failure restores the job; post-delete failure retains intent.

    The kernel answers the purger's ``failed_sync``-th job-root directory sync
    with ``EIO``.
    """
    result = run_child(
        PURGE_PROLOGUE + "job = finished_job('owned evidence')\n"
        "before = manager.record(job)\n"
        "syncs = []\n"
        "def decide(call):\n"
        "    if call.name == 'fsync' and call.descriptor_path(0) == str(root):\n"
        "        syncs.append(1)\n"
        "        if len(syncs) == int(sys.argv[2]):\n"
        "            return errno.EIO\n"
        "    return None\n"
        "hold_system_calls(['fsync'], decide)\n"
        "out = {}\n"
        "try:\n"
        "    manager.purge_terminal_record(job)\n"
        "except OSError as refused:\n"
        "    out['errno'] = refused.errno\n"
        "out['job'] = job\n"
        "out['kept'] = [r.job_id for r in manager.list_records()]\n"
        "out['same'] = bool(out['kept']) and manager.record(job) == before\n"
        "state = manager._ledger.connection().execute('SELECT state FROM job_purges').fetchone()\n"
        "out['state'] = None if state is None else state[0]\n"
        "finish(out)\n",
        arguments=(str(tmp_path / "jobs"), str(failed_sync)),
    )
    job_id = str(result["job"])
    assert result["errno"] == errno.EIO
    manager = reopened(tmp_path / "jobs")
    if failed_sync == 1:
        assert (result["kept"], result["same"], result["state"]) == ([job_id], True, None)
        assert manager.read_artifact(job_id, "proof.txt").payload == b"owned evidence"
        assert manager.status().pending_purge_count == 0
    else:
        assert (result["kept"], result["state"]) == ([], "cleanup_started")
        # The reopened manager has reconciled. Removal happened, but fsync failed
        # before durable completion; absence cannot distinguish this from a
        # moved-away stage after a crash.
        assert manager.list_records() == ()
        assert manager.status().pending_purge_count == 1
        assert (
            manager._ledger.connection().execute("SELECT state FROM job_purges").fetchone()[0]
            == "ambiguous"
        )
        assert not (tmp_path / "jobs" / f".purge-{job_id}").exists()


@_held_calls
def test_purge_syncs_namespace_before_deletion_and_journal_close(tmp_path: Path) -> None:
    """Observe actual directory fsync with committed job/journal state at both boundaries.

    The kernel holds each job-root directory sync, and the first unlink inside
    the stage, while a separate connection reads the committed ledger.
    """
    result = run_child(
        PURGE_PROLOGUE + "import sqlite3\n"
        "job = finished_job('owned evidence')\n"
        "staged = root / ('.purge-' + job)\n"
        "phases, cleanup = [], []\n"
        "def committed():\n"
        "    with sqlite3.connect(manager.ledger_path) as observer:\n"
        "        present = observer.execute('SELECT 1 FROM jobs WHERE job_id=?', (job,)).fetchone()\n"
        "        intent = observer.execute('SELECT state FROM job_purges WHERE job_id=?',\n"
        "            (job,)).fetchone()\n"
        "    return present is not None, None if intent is None else intent[0]\n"
        "def decide(call):\n"
        "    if call.name == 'fsync' and call.descriptor_path(0) == str(root):\n"
        "        phases.append([*committed(), staged.exists()])\n"
        "    if (call.name == 'unlinkat' and not cleanup and call.text(1) == 'proof.txt'\n"
        "            and call.descriptor_path(0) == str(staged)):\n"
        "        cleanup.append(committed()[1])\n"
        "hold_system_calls(['fsync', 'unlinkat'], decide)\n"
        "manager.purge_terminal_record(job)\n"
        "finish({'phases': phases, 'cleanup': cleanup,\n"
        "    'records': len(manager.list_records()),\n"
        "    'pending': manager.status().pending_purge_count})\n",
        arguments=(str(tmp_path / "jobs"),),
    )
    assert result == {
        "phases": [[True, "prepared", True], [False, "cleanup_started", False]],
        "cleanup": ["cleanup_started"],
        "records": 0,
        "pending": 0,
    }


@pytest.mark.parametrize("replacement", [None, "stage", "original"])
def test_restart_closes_committed_removal_evidence(tmp_path: Path, replacement: str | None) -> None:
    """A process dies before journal close; a new manager uses the committed removal phase."""
    manager = StudioJobManager(
        root=tmp_path, allowed_kinds=frozenset({"analysis"}), default_timeout_seconds=3.0
    )
    job = manager.submit(kind="analysis", owner="owner", request_id=None, task=lambda ctx: {})
    assert manager.wait(job.job_id, 2.0).status == "completed"
    assert manager._done_events[job.job_id].wait(1.0)
    inode = (manager.root / job.job_id).stat().st_ino
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import os,sys
from pathlib import Path
from sc_neurocore.studio.platform.jobs import StudioJobManager
m=StudioJobManager(root=Path(sys.argv[1]),allowed_kinds=frozenset({'analysis'}),
                   default_timeout_seconds=3)
c=m._ledger.connection()
c.create_function('interrupt_close',0,lambda:os._exit(71))
c.execute('CREATE TRIGGER stop_before_close BEFORE DELETE ON job_purges '
          'BEGIN SELECT interrupt_close(); END')
m.purge_terminal_record(sys.argv[2])
""",
            str(tmp_path),
            job.job_id,
        ],
        capture_output=True,
        text=True,
        timeout=5.0,
    )
    assert child.returncode == 71, child.stderr
    with sqlite3.connect(manager.ledger_path) as observer:
        assert observer.execute("SELECT state,inode FROM job_purges").fetchall() == [
            ("removed", inode)
        ]
        observer.execute("DROP TRIGGER stop_before_close")
    foreign = None
    if replacement is not None:
        name = f".purge-{job.job_id}" if replacement == "stage" else job.job_id
        directory = tmp_path / name
        directory.mkdir()
        foreign = directory / "unrelated.txt"
        foreign.write_text("foreign custody")
    recovered = StudioJobManager(
        root=tmp_path, allowed_kinds=frozenset({"analysis"}), default_timeout_seconds=3.0
    )
    assert recovered.list_records() == ()
    if foreign is None:
        assert recovered.status().pending_purge_count == 0
        assert not (tmp_path / job.job_id).exists()
        assert not (tmp_path / f".purge-{job.job_id}").exists()
    else:
        recovered.reconcile()
        assert foreign.read_text() == "foreign custody"
        assert recovered.status().pending_purge_count == 1
        assert (
            recovered._ledger.connection().execute("SELECT state FROM job_purges").fetchone()[0]
            == "ambiguous"
        )
