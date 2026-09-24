# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Live-owner purge retry

"""Public reconciliation retries committed cleanup without taking another owner's intent.

Races and refusals are real: the purging interpreter runs as a child whose
chosen system calls the kernel holds while a competing change is made, or
answers with an errno (``tests/studio_syscall_support.py``).
"""

import errno
import sqlite3
import threading
from pathlib import Path

import pytest

from sc_neurocore.studio.platform.jobs import StudioJobManager
from tests.studio_purge_child_support import PURGE_PROLOGUE, reopened
from tests.studio_seccomp_support import SECCOMP_AVAILABLE, run_child

pytestmark = pytest.mark.skipif(not SECCOMP_AVAILABLE, reason="held system calls need Linux x86_64")


def test_purge_retains_intent_when_opened_directory_was_not_removed(tmp_path: Path) -> None:
    """A successful pathname removal cannot certify deletion of a different opened inode.

    The kernel holds the purger's ``rmdir`` of its stage while a real competing
    rename moves the verified directory away and a new one takes its name.
    """
    retained = tmp_path / "retained-original"
    result = run_child(
        PURGE_PROLOGUE + "job = finished_job()\n"
        "original = (root / job).stat().st_ino\n"
        "stage = root / ('.purge-' + job)\n"
        "def decide(call):\n"
        "    if call.name == 'rmdir' and call.text(0) == str(stage):\n"
        "        stage.rename(sys.argv[2])\n"
        "        stage.mkdir()\n"
        "hold_system_calls(['rmdir'], decide)\n"
        "try:\n"
        "    manager.purge_terminal_record(job)\n"
        "    outcome = 'purged'\n"
        "except StudioJobRejected as refused:\n"
        "    outcome = str(refused)\n"
        "finish({'original': original, 'outcome': outcome})\n",
        arguments=(str(tmp_path / "jobs"), str(retained)),
    )
    original_inode = result["original"]
    assert result["outcome"] == "Studio purge cleanup remains pending recovery."
    assert retained.is_dir() and retained.stat().st_ino == original_inode
    manager = reopened(tmp_path / "jobs")
    assert manager.list_records() == ()
    assert manager.status().pending_purge_count == 1
    row = manager._ledger.connection().execute("SELECT * FROM job_purges").fetchone()
    assert row is not None and row["state"] == "ambiguous" and row["inode"] == original_inode
    for _ in range(2):
        manager.reconcile()
        assert manager.status().pending_purge_count == 1
        assert retained.stat().st_ino == original_inode


@pytest.mark.parametrize("remove_original", [False, True])
def test_restart_retains_missing_stage_without_completion_evidence(
    tmp_path: Path, remove_original: bool
) -> None:
    """A dead purger's missing stage cannot distinguish unlink from displaced custody.

    While the kernel holds the purger's ``rmdir``, the stage is removed or
    displaced for real and the purger is killed before the call returns.
    """
    root = tmp_path / "jobs"
    retained = tmp_path / "retained-original"
    manager = reopened(root)
    job = manager.submit(kind="analysis", owner="owner", request_id=None, task=lambda ctx: {})
    assert manager.wait(job.job_id, 10.0).status == "completed"
    assert manager._done_events[job.job_id].wait(5.0)
    original_inode = (root / job.job_id).stat().st_ino
    run_child(
        PURGE_PROLOGUE + "job, retained = sys.argv[2], Path(sys.argv[3])\n"
        "stage = root / ('.purge-' + job)\n"
        "def decide(call):\n"
        "    if call.name == 'rmdir' and call.text(0) == str(stage):\n"
        "        if sys.argv[4] == 'True':\n"
        "            os.rmdir(stage)\n"
        "        else:\n"
        "            stage.rename(retained)\n"
        "        os.kill(os.getpid(), signal.SIGKILL)\n"
        "hold_system_calls(['rmdir'], decide)\n"
        "manager.purge_terminal_record(job)\n",
        arguments=(str(root), job.job_id, str(retained), str(remove_original)),
        expected_returncode=-9,
    )
    assert not (root / f".purge-{job.job_id}").exists()
    for _ in range(2):
        recovered = reopened(root)
        recovered.reconcile()
        assert recovered.list_records() == ()
        assert recovered.status().pending_purge_count == 1
        row = recovered._ledger.connection().execute("SELECT * FROM job_purges").fetchone()
        assert row is not None and row["inode"] == original_inode
        assert row["state"] == "ambiguous"
    if not remove_original:
        assert retained.stat().st_ino == original_inode


def test_missing_directory_purge_refusal_retains_record_and_allows_retry(tmp_path: Path) -> None:
    """A refused database deletion leaves history intact even without a directory to restore."""
    manager = StudioJobManager(
        root=tmp_path / "jobs",
        allowed_kinds=frozenset({"analysis"}),
        default_timeout_seconds=3.0,
    )
    job = manager.submit(kind="analysis", owner="owner", request_id=None, task=lambda ctx: {})
    assert manager.wait(job.job_id, 2.0).status == "completed"
    assert manager._done_events[job.job_id].wait(1.0)
    record = manager.record(job.job_id)
    history = manager.transitions(job.job_id)
    original = manager.root / job.job_id
    preserved = tmp_path / "retained-directory"
    original.rename(preserved)
    with manager._ledger.transaction() as connection:
        connection.execute(
            "CREATE TRIGGER refuse_missing_purge BEFORE DELETE ON jobs "
            "BEGIN SELECT RAISE(ABORT,'injected missing-directory refusal'); END"
        )
    with pytest.raises(sqlite3.IntegrityError, match="missing-directory refusal"):
        manager.purge_terminal_record(job.job_id)
    assert manager.record(job.job_id) == record
    assert manager.transitions(job.job_id) == history
    assert manager.status().pending_purge_count == 0
    assert not original.exists()
    assert not (manager.root / f".purge-{job.job_id}").exists()
    assert preserved.is_dir()
    with manager._ledger.transaction() as connection:
        connection.execute("DROP TRIGGER refuse_missing_purge")
    assert manager.purge_terminal_record(job.job_id) == record
    assert manager.list_records() == ()
    assert manager.status().pending_purge_count == 0
    assert job.job_id not in manager._done_events
    assert job.job_id not in manager._cancel_events
    assert preserved.is_dir()


@pytest.mark.parametrize("clear_before_error", [False, True])
def test_reconcile_retries_own_committed_cleanup(tmp_path: Path, clear_before_error: bool) -> None:
    """A transient filesystem refusal does not require restarting the live supervisor.

    The kernel refuses the stage's evidence unlink, or the emptied stage's
    ``rmdir``, until the refusal is lifted; a peer process never takes the
    purge over.
    """
    result = run_child(
        PURGE_PROLOGUE + "import subprocess\n"
        "job = finished_job('owned cleanup evidence')\n"
        "stage = root / ('.purge-' + job)\n"
        "refusing = [True]\n"
        "emptied = sys.argv[2] == 'True'\n"
        "def decide(call):\n"
        "    if not refusing[0]:\n"
        "        return None\n"
        "    if emptied and call.name == 'rmdir' and call.text(0) == str(stage):\n"
        "        return errno.EBUSY\n"
        "    if (not emptied and call.name == 'unlinkat' and call.text(1) == 'proof.txt'\n"
        "            and call.descriptor_path(0) == str(stage)):\n"
        "        return errno.EACCES\n"
        "    return None\n"
        "hold_system_calls(['rmdir', 'unlinkat'], decide)\n"
        "out = {}\n"
        "try:\n"
        "    manager.purge_terminal_record(job)\n"
        "except OSError as refused:\n"
        "    out['purge'] = refused.errno\n"
        "out['records'] = len(manager.list_records())\n"
        "out['pending'] = manager.status().pending_purge_count\n"
        "out['remaining'] = sorted(os.listdir(stage))\n"
        "peer = subprocess.run([sys.executable, '-c', 'from pathlib import Path; import sys; '\n"
        "    'from sc_neurocore.studio.platform.jobs import StudioJobManager; '\n"
        "    'm=StudioJobManager(root=Path(sys.argv[1]),allowed_kinds=frozenset({\"analysis\"}),'\n"
        "    'default_timeout_seconds=3); m.reconcile(); '\n"
        "    'assert m.status().pending_purge_count == 1', str(root)])\n"
        "out['peer'] = peer.returncode\n"
        "out['after_peer'] = sorted(os.listdir(stage))\n"
        "try:\n"
        "    manager.reconcile()\n"
        "except OSError as refused:\n"
        "    out['reconcile'] = refused.errno\n"
        "out['pending_refused'] = manager.status().pending_purge_count\n"
        "refusing[0] = False\n"
        "manager.reconcile()\n"
        "out['pending_final'] = manager.status().pending_purge_count\n"
        "out['stage'] = stage.exists()\n"
        "out['records_final'] = len(manager.list_records())\n"
        "manager.reconcile()\n"
        "out['pending_again'] = manager.status().pending_purge_count\n"
        "finish(out)\n",
        arguments=(str(tmp_path / "jobs"), str(clear_before_error)),
    )
    refusal = errno.EBUSY if clear_before_error else errno.EACCES
    remaining = [] if clear_before_error else ["proof.txt"]
    assert result == {
        "purge": refusal,
        "records": 0,
        "pending": 1,
        "remaining": remaining,
        "peer": 0,
        "after_peer": remaining,
        "reconcile": refusal,
        "pending_refused": 1,
        "pending_final": 0,
        "stage": False,
        "records_final": 0,
        "pending_again": 0,
    }


def test_reconcile_after_delete_commit_does_not_make_purge_report_failure(
    tmp_path: Path,
) -> None:
    """A recovery pass between commit and cleanup is a valid completion, not pending work.

    When the purger's connection starts its first statement after the deletion
    committed, a real reconcile on another thread and connection completes the
    cleanup before the purger continues.
    """
    manager = reopened(tmp_path / "jobs")
    job = manager.submit(kind="analysis", owner="owner", request_id=None, task=lambda ctx: {})
    assert manager.wait(job.job_id, 10.0).status == "completed"
    assert manager._done_events[job.job_id].wait(5.0)
    record = manager.record(job.job_id)
    statements: list[str] = []
    observed: list[int] = []

    def reconcile_elsewhere() -> None:
        try:
            manager.reconcile()
            observed.append(manager.status().pending_purge_count)
        finally:
            manager._ledger.close()

    def trace(statement: str) -> None:
        committed = any("state='committed'" in text for text in statements)
        if committed and statements[-1] == "COMMIT" and not observed:
            other = threading.Thread(target=reconcile_elsewhere)
            other.start()
            other.join(timeout=30.0)
        statements.append(statement)

    connection = manager._ledger.connection()
    connection.set_trace_callback(trace)
    try:
        assert manager.purge_terminal_record(job.job_id) == record
    finally:
        connection.set_trace_callback(None)
    assert observed == [0]
    assert manager.list_records() == ()
    assert manager.status().pending_purge_count == 0
    assert not (tmp_path / "jobs" / f".purge-{job.job_id}").exists()
    assert job.job_id not in manager._done_events
    assert job.job_id not in manager._cancel_events
