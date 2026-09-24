# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Purge commit ambiguity

"""Recover from reported commit errors using persisted phase, not exception wording."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from sc_neurocore.studio.platform.jobs import StudioJobManager
from tests.studio_purge_child_support import PURGE_PROLOGUE
from tests.studio_seccomp_support import SECCOMP_AVAILABLE, run_child


def test_finishing_one_purge_does_not_recover_another_intent(tmp_path: Path) -> None:
    """Exact-operation cleanup does not silently perform another pending purge recovery."""
    manager = StudioJobManager(
        root=tmp_path,
        allowed_kinds=frozenset({"analysis"}),
        default_timeout_seconds=3.0,
    )
    jobs = [
        manager.submit(kind="analysis", owner="owner", request_id=None, task=lambda ctx: {})
        for _ in range(2)
    ]
    for job in jobs:
        assert manager.wait(job.job_id, 2.0).status == "completed"
        assert manager._done_events[job.job_id].wait(1.0)
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            "from sc_neurocore.studio.platform.jobs_ledger_supervisor import supervisor_identity; print(supervisor_identity())",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=3.0,
    )
    other = jobs[1].job_id
    stat = (tmp_path / other).stat()
    with manager._ledger.transaction() as connection:
        connection.execute(
            "INSERT INTO job_purges VALUES(?,?,?,?, 'prepared')",
            (other, child.stdout.strip(), stat.st_dev, stat.st_ino),
        )
    manager.purge_terminal_record(jobs[0].job_id)
    assert manager.record(other).status == "completed"
    assert (
        manager._ledger.connection().execute("SELECT job_id FROM job_purges").fetchone()[0] == other
    )


@pytest.mark.skipif(not SECCOMP_AVAILABLE, reason="held system calls need Linux x86_64")
@pytest.mark.parametrize("committed", [False, True, "open"])
@pytest.mark.parametrize("phase", [1, 2])
def test_purge_commit_error_preserves_actual_database_outcome(
    tmp_path: Path, committed: bool | str, phase: int
) -> None:
    """Fail or interrupt a real SQLite commit and verify public custody and local handles.

    For the ``phase``-th purge transaction the kernel answers the WAL's commit
    sync with ``EIO`` (not committed); or a real signal interrupts the
    purger right after that durable commit (committed, acknowledgement lost);
    or right after ``BEGIN IMMEDIATE`` took the WAL write lock, before the
    transaction body (left open).
    """
    result = run_child(
        PURGE_PROLOGUE + "import sqlite3\n"
        "job = finished_job('commit boundary evidence')\n"
        "before = manager.record(job)\n"
        "mode, phase = sys.argv[2], int(sys.argv[3])\n"
        "def lost(number, frame):\n"
        "    raise sqlite3.OperationalError('injected commit acknowledgement failure')\n"
        "signal.signal(signal.SIGUSR1, lost)\n"
        "begins, commits = [], []\n"
        "def decide(call):\n"
        "    if (call.name == 'fcntl' and call.arguments[1] == 6\n"
        "            and call.descriptor_path(0).endswith('-shm')):\n"
        "        lock = call.memory(2, 16)\n"
        "        if lock[0:2] == b'\\x01\\x00' and int.from_bytes(lock[8:16], 'little') == 120:\n"
        "            begins.append(1)\n"
        "            if mode == 'open' and len(begins) == phase:\n"
        "                os.kill(os.getpid(), signal.SIGUSR1)\n"
        "    if call.name in ('fsync', 'fdatasync') and call.descriptor_path(0).endswith('-wal'):\n"
        "        commits.append(1)\n"
        "        if len(commits) == phase and mode == 'False':\n"
        "            return errno.EIO\n"
        "        if len(commits) == phase and mode == 'True':\n"
        "            os.kill(os.getpid(), signal.SIGUSR1)\n"
        "    return None\n"
        "hold_system_calls(['fcntl', 'fsync', 'fdatasync'], decide)\n"
        "# Only the notifier may take the signal, so no held call is interrupted.\n"
        "signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGUSR1})\n"
        "try:\n"
        "    manager.purge_terminal_record(job)\n"
        "    raised = None\n"
        "except sqlite3.OperationalError as refused:\n"
        "    raised = str(refused)\n"
        "out = {'job': job, 'raised': raised,\n"
        "    'in_transaction': manager._ledger.connection().in_transaction}\n"
        "with sqlite3.connect(manager.ledger_path, timeout=0.2) as observer:\n"
        "    observer.execute('BEGIN IMMEDIATE')\n"
        "    out['intents'] = observer.execute('SELECT COUNT(*) FROM job_purges').fetchone()[0]\n"
        "    observer.rollback()\n"
        "out['stage'] = (root / ('.purge-' + job)).exists()\n"
        "out['directory'] = (root / job).exists()\n"
        "out['kept'] = [r.job_id for r in manager.list_records()] == [job]\n"
        "out['same'] = out['kept'] and manager.record(job) == before\n"
        "out['readable'] = out['kept'] and manager.read_artifact(job, 'proof.txt').payload.decode()\n"
        "out['handles'] = job in manager._done_events or job in manager._cancel_events\n"
        "finish(out)\n",
        arguments=(str(tmp_path), str(committed), str(phase)),
    )
    expected_error = (
        "disk I/O error" if committed is False else "injected commit acknowledgement failure"
    )
    purged = committed is True and phase == 2
    assert result == {
        "job": result["job"],
        "raised": expected_error,
        "in_transaction": False,
        "intents": 0,
        "stage": False,
        "directory": not purged,
        "kept": not purged,
        "same": not purged,
        "readable": False if purged else "commit boundary evidence",
        "handles": not purged,
    }
