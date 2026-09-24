# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Purge recovery path conflicts

"""Recover only recorded directory custody, never a replacement at the same path."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from sc_neurocore.studio.platform.jobs import StudioJobContext, StudioJobManager
from sc_neurocore.studio.platform.jobs_models import StudioJobRejected
from tests.studio_purge_child_support import PURGE_PROLOGUE
from tests.studio_seccomp_support import SECCOMP_AVAILABLE, run_child

_held_calls = pytest.mark.skipif(
    not SECCOMP_AVAILABLE, reason="held system calls need Linux x86_64"
)


@_held_calls
@pytest.mark.parametrize("after_open", [False, True])
@pytest.mark.parametrize("replacement", ["directory", "missing", "file", "symlink"])
def test_purge_rechecks_opened_directory_after_identity_race(
    tmp_path: Path, after_open: bool, replacement: str
) -> None:
    """Replacing a verified pathname must not authorize deleting the new directory.

    The kernel holds the purger right after it verified the stage (its parent
    sync open) or right after it opened the stage (its first directory read)
    while the stage is replaced for real.
    """
    root = tmp_path / "jobs"
    retained = tmp_path / "retained-stage"
    external = tmp_path / "external-data"
    external.mkdir()
    (external / "foreign.txt").write_text("external evidence")
    result = run_child(
        PURGE_PROLOGUE + "job = finished_job()\n"
        "(root / job / 'owned.txt').write_text('original evidence')\n"
        "stage = root / ('.purge-' + job)\n"
        "retained, external = Path(sys.argv[2]), Path(sys.argv[3])\n"
        "replacement, after_open = sys.argv[4], sys.argv[5] == 'True'\n"
        "replaced = []\n"
        "def replace_stage():\n"
        "    replaced.append(True)\n"
        "    stage.rename(retained)\n"
        "    if replacement == 'directory':\n"
        "        stage.mkdir()\n"
        "        (stage / 'foreign.txt').write_text('not authorized for deletion')\n"
        "    elif replacement == 'file':\n"
        "        stage.write_text('replacement file')\n"
        "    elif replacement == 'symlink':\n"
        "        stage.symlink_to(external, target_is_directory=True)\n"
        "def decide(call):\n"
        "    if replaced or not stage.exists():\n"
        "        return None\n"
        "    if after_open and call.name == 'getdents64':\n"
        "        if call.descriptor_path(0) == str(stage):\n"
        "            replace_stage()\n"
        "    elif not after_open and call.name == 'openat' and call.text(1) == str(root):\n"
        "        replace_stage()\n"
        "hold_system_calls(['openat', 'getdents64'], decide)\n"
        "try:\n"
        "    manager.purge_terminal_record(job)\n"
        "    rejected = False\n"
        "except StudioJobRejected:\n"
        "    rejected = True\n"
        "finish({'replaced': replaced, 'rejected': rejected,\n"
        "    'pending': manager.status().pending_purge_count})\n",
        arguments=(str(root), str(retained), str(external), replacement, str(after_open)),
    )
    assert result == {"replaced": [True], "rejected": True, "pending": 1}
    stage = next(root.glob(".purge-*"), root / ".purge-missing")
    assert (external / "foreign.txt").read_text() == "external evidence"
    if replacement == "directory":
        assert (stage / "foreign.txt").read_text() == "not authorized for deletion"
    elif replacement == "file":
        assert stage.read_text() == "replacement file"
    elif replacement == "symlink":
        assert stage.is_symlink() and stage.resolve() == external
    else:
        assert not stage.exists()
    if after_open:
        assert retained.is_dir()
        assert not (retained / "owned.txt").exists()
    else:
        assert (retained / "owned.txt").read_text() == "original evidence"


@_held_calls
@pytest.mark.parametrize("acknowledgement_error", [False, True])
def test_postcommit_stage_replacement_reports_pending_cleanup(
    tmp_path: Path, acknowledgement_error: bool
) -> None:
    """Committed deletion does not authorize erasing a different staged directory.

    After the deletion committed, the kernel holds the purger's first open of
    its stage while the stage is replaced for real; a real asynchronous signal
    can also interrupt the purger there, as a lost acknowledgement would.
    """
    root = tmp_path / "jobs"
    retained = tmp_path / "original-stage"
    result = run_child(
        PURGE_PROLOGUE + "job = finished_job()\n"
        "stage = root / ('.purge-' + job)\n"
        "retained, interrupt = Path(sys.argv[2]), sys.argv[3] == 'True'\n"
        "def lost(number, frame):\n"
        "    raise RuntimeError('commit acknowledgement lost')\n"
        "signal.signal(signal.SIGUSR1, lost)\n"
        "replaced = []\n"
        "def decide(call):\n"
        "    if call.name == 'openat' and not replaced and call.text(1) == str(stage):\n"
        "        replaced.append(True)\n"
        "        stage.rename(retained)\n"
        "        stage.mkdir()\n"
        "        (stage / 'foreign.txt').write_text('not part of this purge')\n"
        "        if interrupt:\n"
        "            os.kill(os.getpid(), signal.SIGUSR1)\n"
        "hold_system_calls(['openat'], decide)\n"
        "try:\n"
        "    manager.purge_terminal_record(job)\n"
        "    raised = None\n"
        "except (RuntimeError, StudioJobRejected) as refused:\n"
        "    raised = [type(refused).__name__, str(refused)]\n"
        "finish({'job': job, 'replaced': replaced, 'raised': raised,\n"
        "    'records': len(manager.list_records()),\n"
        "    'pending': manager.status().pending_purge_count,\n"
        "    'forgotten': job not in manager._done_events and job not in manager._cancel_events})\n",
        arguments=(str(root), str(retained), str(acknowledgement_error)),
    )
    expected = (
        ["RuntimeError", "commit acknowledgement lost"]
        if acknowledgement_error
        else ["StudioJobRejected", "Studio purge cleanup remains pending recovery."]
    )
    assert result == {
        "job": result["job"],
        "replaced": [True],
        "raised": expected,
        "records": 0,
        "pending": 1,
        "forgotten": True,
    }
    assert retained.is_dir()
    stage = root / f".purge-{result['job']}"
    assert (stage / "foreign.txt").read_text() == "not part of this purge"


def test_purge_clears_nested_owned_files_without_following_links(tmp_path: Path) -> None:
    """Descriptor-relative cleanup removes nested artifacts but preserves external data."""
    manager = StudioJobManager(
        root=tmp_path / "jobs",
        allowed_kinds=frozenset({"analysis"}),
        default_timeout_seconds=3.0,
    )

    def task(context: StudioJobContext) -> dict[str, object]:
        context.write_artifact("nested/proof.txt", "owned nested evidence")
        return {}

    job = manager.submit(kind="analysis", owner="owner", request_id=None, task=task)
    assert manager.wait(job.job_id, 2.0).status == "completed"
    assert manager._done_events[job.job_id].wait(1.0)
    external = tmp_path / "external"
    external.mkdir()
    (external / "proof.txt").write_text("external evidence")
    original = manager.root / job.job_id
    (original / "outside").symlink_to(external, target_is_directory=True)
    manager.purge_terminal_record(job.job_id)
    assert not original.exists()
    assert not (manager.root / f".purge-{job.job_id}").exists()
    assert manager.list_records() == ()
    assert manager.status().pending_purge_count == 0
    assert (external / "proof.txt").read_text() == "external evidence"


@pytest.mark.parametrize("shape", ["missing", "file", "symlink"])
def test_purge_handles_missing_or_invalid_directory(tmp_path: Path, shape: str) -> None:
    """An invalid target is refused without deleting replacement or retained bytes."""
    manager = StudioJobManager(
        root=tmp_path / "jobs",
        allowed_kinds=frozenset({"analysis"}),
        default_timeout_seconds=3.0,
    )

    def task(context: StudioJobContext) -> dict[str, object]:
        context.write_artifact("proof.txt", "retained job evidence")
        return {}

    job = manager.submit(kind="analysis", owner="owner", request_id=None, task=task)
    assert manager.wait(job.job_id, 2.0).status == "completed"
    assert manager._done_events[job.job_id].wait(1.0)
    record = manager.record(job.job_id)
    original = manager.root / job.job_id
    retained = tmp_path / "retained-original"
    original.rename(retained)
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "proof.txt").write_text("foreign evidence")
    if shape == "file":
        original.write_text("replacement file")
    elif shape == "symlink":
        original.symlink_to(foreign, target_is_directory=True)
    if shape == "missing":
        assert manager.purge_terminal_record(job.job_id) == record
        assert manager.list_records() == ()
    else:
        with pytest.raises(StudioJobRejected, match="directory|symlink"):
            manager.purge_terminal_record(job.job_id)
        assert manager.record(job.job_id) == record
        if shape == "file":
            assert original.read_text() == "replacement file"
        else:
            assert original.is_symlink()
    assert (retained / "proof.txt").read_text() == "retained job evidence"
    assert (foreign / "proof.txt").read_text() == "foreign evidence"
    assert (
        manager._ledger.connection().execute("SELECT COUNT(*) FROM job_purges").fetchone()[0] == 0
    )


@pytest.mark.parametrize("phase", ["prepared", "committed"])
@pytest.mark.parametrize("conflict", ["replacement", "symlink", "original"])
def test_recovery_preserves_conflicting_paths(tmp_path: Path, phase: str, conflict: str) -> None:
    """An actual dead owner does not authorize deleting unrelated directory contents."""
    manager = StudioJobManager(
        root=tmp_path / "jobs",
        allowed_kinds=frozenset({"analysis"}),
        default_timeout_seconds=3.0,
    )
    job = manager.submit(kind="analysis", owner="owner", request_id=None, task=lambda ctx: {})
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
    original = manager.root / job.job_id
    stage = manager.root / f".purge-{job.job_id}"
    (original / "retained.txt").write_text("owned evidence")
    stat = original.stat()
    original.rename(stage)
    if phase == "committed":
        manager._ledger.delete(job.job_id)
    with manager._ledger.transaction() as connection:
        connection.execute(
            "INSERT INTO job_purges VALUES(?,?,?,?,?)",
            (job.job_id, child.stdout.strip(), stat.st_dev, stat.st_ino, phase),
        )
    if conflict == "original":
        original.mkdir()
        foreign = original / "foreign.txt"
        foreign.write_text("unrelated evidence")
        retained = stage
    else:
        retained = tmp_path / "preserved-stage"
        stage.rename(retained)
        if conflict == "replacement":
            stage.mkdir()
            foreign = stage / "foreign.txt"
        else:
            external = tmp_path / "unrelated-directory"
            external.mkdir()
            stage.symlink_to(external, target_is_directory=True)
            foreign = external / "foreign.txt"
        foreign.write_text("unrelated evidence")
    records = manager.list_records()
    journal = tuple(manager._ledger.connection().execute("SELECT * FROM job_purges").fetchone())
    manager.reconcile()
    assert foreign.read_text() == "unrelated evidence"
    assert (retained / "retained.txt").read_text() == "owned evidence"
    assert manager.list_records() == records
    assert tuple(manager._ledger.connection().execute("SELECT * FROM job_purges").fetchone()) == (
        *journal[:4],
        "ambiguous" if phase == "committed" else phase,
    )
    if conflict == "symlink":
        assert stage.is_symlink()
