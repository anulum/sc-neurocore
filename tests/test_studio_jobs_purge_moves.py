# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Purge namespace move custody

"""Restoration must not overwrite a target created after the absence check."""

from pathlib import Path
import subprocess
import sys
import errno

import pytest

from sc_neurocore.studio.platform.jobs import StudioJobContext
from tests.studio_purge_child_support import PURGE_PROLOGUE, reopened
from tests.studio_seccomp_support import SECCOMP_AVAILABLE, run_child

_held_calls = pytest.mark.skipif(
    not SECCOMP_AVAILABLE, reason="held system calls need Linux x86_64"
)


_REFUSED_RENAME = (
    "refusing = [int(sys.argv[2])]\n"
    "def decide(call):\n"
    "    if refusing[0] and call.name == 'renameat2':\n"
    "        return refusing[0]\n"
    "    return None\n"
    "def snapshot(job):\n"
    "    return [manager.record(job).to_public_dict(),\n"
    "        [dict(t) for t in manager.transitions(job)]]\n"
)


@_held_calls
@pytest.mark.parametrize("error", [errno.EOPNOTSUPP, errno.EACCES, errno.EIO])
def test_native_restore_refusal_retains_journal_for_retry(tmp_path: Path, error: int) -> None:
    """Already moved artifacts remain recoverable after the kernel refuses restoration."""
    result = run_child(
        PURGE_PROLOGUE + _REFUSED_RENAME + "import subprocess\n"
        "from sc_neurocore.studio.platform.jobs_purge_paths import move_without_replace\n"
        "job = finished_job('recoverable evidence')\n"
        "before = snapshot(job)\n"
        "departed = subprocess.run([sys.executable, '-c', 'from sc_neurocore.studio.platform.'\n"
        "    'jobs_ledger_supervisor import supervisor_identity; print(supervisor_identity())'],\n"
        "    capture_output=True, text=True, check=True).stdout.strip()\n"
        "original, stage = root / job, root / ('.purge-' + job)\n"
        "identity = original.stat()\n"
        "with manager._ledger.transaction() as connection:\n"
        "    connection.execute(\"INSERT INTO job_purges VALUES(?,?,?,?, 'prepared')\",\n"
        "        (job, departed, identity.st_dev, identity.st_ino))\n"
        "assert move_without_replace(original, stage)\n"
        "journal = list(manager._ledger.connection().execute('SELECT * FROM job_purges').fetchone())\n"
        "hold_system_calls(['renameat2'], decide)\n"
        "out = {}\n"
        "try:\n"
        "    manager.reconcile()\n"
        "except OSError as refused:\n"
        "    out['errno'] = refused.errno\n"
        "out['original'] = original.exists()\n"
        "out['evidence'] = (stage / 'proof.txt').read_text()\n"
        "out['unchanged'] = snapshot(job) == before\n"
        "out['journal'] = list(manager._ledger.connection().execute(\n"
        "    'SELECT * FROM job_purges').fetchone()) == journal\n"
        "out['pending'] = manager.status().pending_purge_count\n"
        "refusing[0] = 0\n"
        "manager.reconcile()\n"
        "out['restored'] = manager.read_artifact(job, 'proof.txt').payload.decode()\n"
        "out['after'] = snapshot(job) == before\n"
        "out['stage'] = stage.exists()\n"
        "out['pending_final'] = manager.status().pending_purge_count\n"
        "finish(out)\n",
        arguments=(str(tmp_path / "jobs"), str(error)),
    )
    assert result == {
        "errno": error,
        "original": False,
        "evidence": "recoverable evidence",
        "unchanged": True,
        "journal": True,
        "pending": 1,
        "restored": "recoverable evidence",
        "after": True,
        "stage": False,
        "pending_final": 0,
    }


@_held_calls
@pytest.mark.parametrize("error", [errno.EOPNOTSUPP, errno.EACCES, errno.EIO])
def test_failed_native_staging_preserves_readable_job(tmp_path: Path, error: int) -> None:
    """A refused atomic move never falls back to a destructive non-atomic one.

    A C library without ``renameat2`` is a platform outside the supported set;
    its ``ENOSYS`` branch stays visibly uncovered.
    """
    result = run_child(
        PURGE_PROLOGUE + _REFUSED_RENAME + "job = finished_job('owned evidence')\n"
        "before = snapshot(job)\n"
        "hold_system_calls(['renameat2'], decide)\n"
        "out = {}\n"
        "try:\n"
        "    manager.purge_terminal_record(job)\n"
        "except OSError as refused:\n"
        "    out['errno'] = refused.errno\n"
        "out['unchanged'] = snapshot(job) == before\n"
        "out['readable'] = manager.read_artifact(job, 'proof.txt').payload.decode()\n"
        "out['stage'] = (root / ('.purge-' + job)).exists()\n"
        "out['pending'] = manager.status().pending_purge_count\n"
        "refusing[0] = 0\n"
        "out['purged'] = manager.purge_terminal_record(job).job_id == job\n"
        "out['records'] = len(manager.list_records())\n"
        "finish(out)\n",
        arguments=(str(tmp_path / "jobs"), str(error)),
    )
    assert result == {
        "errno": error,
        "unchanged": True,
        "readable": "owned evidence",
        "stage": False,
        "pending": 0,
        "purged": True,
        "records": 0,
    }


@_held_calls
def test_staging_does_not_replace_new_empty_target(tmp_path: Path) -> None:
    """A destination appearing after its last absence check remains untouched.

    The kernel holds the staging rename while a real directory takes the
    stage name; the non-replacing rename then refuses.
    """
    root = tmp_path / "jobs"
    result = run_child(
        PURGE_PROLOGUE + "job = finished_job('owned evidence')\n"
        "stage = root / ('.purge-' + job)\n"
        "created = []\n"
        "def decide(call):\n"
        "    if call.name == 'renameat2' and call.text(3) == str(stage) and not created:\n"
        "        stage.mkdir()\n"
        "        created.append(stage.stat().st_ino)\n"
        "hold_system_calls(['renameat2'], decide)\n"
        "try:\n"
        "    manager.purge_terminal_record(job)\n"
        "    rejected = False\n"
        "except StudioJobRejected:\n"
        "    rejected = True\n"
        "finish({'job': job, 'created': created, 'rejected': rejected})\n",
        arguments=(str(root),),
    )
    job_id = str(result["job"])
    stage = root / f".purge-{job_id}"
    assert result["rejected"] is True
    assert stage.is_dir(), "Purge overwrote and erased an unrelated staging directory"
    assert [stage.stat().st_ino] == result["created"]
    assert (root / job_id / "proof.txt").read_text() == "owned evidence"
    manager = reopened(root)
    assert [record.job_id for record in manager.list_records()] == [job_id]
    assert manager.status().pending_purge_count == 1


@_held_calls
def test_recovery_does_not_replace_new_empty_target(tmp_path: Path) -> None:
    """Retain both directories and pending intent when the restore target appears.

    A departed supervisor's prepared purge is recovered by a new manager; the
    kernel holds its restoring rename while a real directory takes the
    original name.
    """
    root = tmp_path / "jobs"
    manager = reopened(root)

    def task(context: StudioJobContext) -> dict[str, object]:
        context.write_artifact("proof.txt", "owned evidence")
        return {}

    job = manager.submit(kind="analysis", owner="owner", request_id=None, task=task)
    assert manager.wait(job.job_id, 10.0).status == "completed"
    assert manager._done_events[job.job_id].wait(5.0)
    departed = subprocess.run(
        [
            sys.executable,
            "-c",
            "from sc_neurocore.studio.platform.jobs_ledger_supervisor "
            "import supervisor_identity; print(supervisor_identity())",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30.0,
    ).stdout.strip()
    original = root / job.job_id
    stage = root / f".purge-{job.job_id}"
    identity = original.stat()
    original.rename(stage)
    with manager._ledger.transaction() as connection:
        connection.execute(
            "INSERT INTO job_purges VALUES(?,?,?,?, 'prepared')",
            (job.job_id, departed, identity.st_dev, identity.st_ino),
        )
    records = manager.list_records()
    manager._ledger.close()
    result = run_child(
        "import json, sys\n"
        "from pathlib import Path\n"
        "from sc_neurocore.studio.platform.jobs import StudioJobManager\n"
        "from tests.studio_syscall_support import finish, hold_system_calls\n"
        "root, original = Path(sys.argv[1]), Path(sys.argv[2])\n"
        "created = []\n"
        "def decide(call):\n"
        "    if call.name == 'renameat2' and call.text(3) == str(original) and not created:\n"
        "        original.mkdir()\n"
        "        created.append(original.stat().st_ino)\n"
        "hold_system_calls(['renameat2'], decide)\n"
        "manager = StudioJobManager(root=root, allowed_kinds=frozenset({'analysis'}),\n"
        "    default_timeout_seconds=3.0)\n"
        "manager.reconcile()\n"
        "finish({'created': created})\n",
        arguments=(str(root), str(original)),
    )
    assert [original.stat().st_ino] == result["created"], (
        "Recovery replaced an unrelated empty directory"
    )
    assert list(original.iterdir()) == []
    assert (stage / "proof.txt").read_text() == "owned evidence"
    recovered = reopened(root)
    assert recovered.list_records() == records
    assert recovered.status().pending_purge_count == 1
