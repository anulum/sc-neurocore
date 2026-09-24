# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Capacity recovery after actual supervisor death

"""Recover slots only when actual process death proves the work is gone."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from sc_neurocore.studio.platform.jobs import StudioJobManager
from sc_neurocore.studio.platform.jobs_admission import StudioJobQueueFull
from sc_neurocore.studio.platform.jobs_process_state import group_survivors
from sc_neurocore.studio.platform.jobs_models import StudioJobRejected


@pytest.mark.parametrize("metadata", ["valid", "missing", "foreign", "bad-boot", "wrong-owner"])
def test_dead_supervisor_does_not_release_a_live_orphan_process(
    tmp_path: Path, metadata: str
) -> None:
    """A process worker can survive its supervisor; recovery must retain its slot."""
    manager = StudioJobManager(
        root=tmp_path,
        allowed_kinds=frozenset({"analysis"}),
        default_timeout_seconds=10.0,
        max_concurrent_jobs=1,
        max_queued_jobs=0,
    )
    worker_pid: int | None = None
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            """
import sys,threading
from pathlib import Path
from sc_neurocore.studio.platform.jobs import StudioJobManager
m=StudioJobManager(root=Path(sys.argv[1]),allowed_kinds=frozenset({'analysis'}),
    default_timeout_seconds=20.0,max_concurrent_jobs=1,max_queued_jobs=0)
m.submit_process_task(kind='analysis',owner='child',request_id=None,
    task_path='tests.test_studio_jobs_cancel_race:peer_cancel_process_task',payload={})
threading.Event().wait(30)
""",
            str(tmp_path),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 5.0
        markers = list(tmp_path.glob("sj_*/worker.pid"))
        while not markers and time.monotonic() < deadline:
            assert child.poll() is None
            time.sleep(0.01)
            markers = list(tmp_path.glob("sj_*/worker.pid"))
        assert len(markers) == 1
        worker_pid = int(markers[0].read_text())
        os.kill(worker_pid, 0)
        # Fault-inject an unschedulable group, including its independent guard.
        # Recovery must retain custody even when automatic cleanup cannot run.
        os.killpg(worker_pid, signal.SIGSTOP)
        child.kill()
        child.wait(timeout=3.0)
        decisions = manager.reconcile()
        assert len(decisions) == 1 and decisions[0].status == "interrupted"
        os.kill(worker_pid, 0)
        assert manager._admission.snapshot().running == 1
        retained = manager.list_records()
        with pytest.raises(StudioJobRejected, match="capacity"):
            manager.purge_terminal_record(decisions[0].job_id)
        with pytest.raises(StudioJobRejected, match="capacity"):
            manager._ledger.delete(decisions[0].job_id)
        assert manager.list_records() == retained
        assert markers[0].is_file()
        with pytest.raises(StudioJobQueueFull):
            manager.submit(kind="analysis", owner="observer", request_id=None, task=lambda ctx: {})
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=3.0)
        if worker_pid is None:
            markers = list(tmp_path.glob("sj_*/worker.pid"))
            if len(markers) == 1:
                worker_pid = int(markers[0].read_text())
        if worker_pid is not None:
            try:
                os.killpg(worker_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            deadline = time.monotonic() + 3.0
            while group_survivors(worker_pid) and time.monotonic() < deadline:
                time.sleep(0.01)
            assert not group_survivors(worker_pid), "Test worker group survived cleanup"
    before = manager.list_records()
    with manager._ledger.transaction() as connection:
        if metadata == "missing":
            connection.execute("DELETE FROM job_workers")
        elif metadata == "foreign":
            connection.execute("UPDATE job_workers SET worker_identity='foreign-host:1:1'")
        elif metadata == "bad-boot":
            connection.execute("UPDATE job_workers SET boot_id='not-a-boot-id'")
        elif metadata == "wrong-owner":
            connection.execute("UPDATE job_workers SET supervisor='different-owner'")
    manager.reconcile()
    assert manager._admission.snapshot().running == (0 if metadata == "valid" else 1)
    assert manager.list_records() == before
    if metadata == "valid":
        job_id = before[0].job_id
        assert manager.purge_terminal_record(job_id) == before[0]
        assert not (tmp_path / job_id).exists()
        assert manager.list_records() == ()
        assert (
            manager._ledger.connection()
            .execute("SELECT COUNT(*) FROM job_workers WHERE job_id=?", (job_id,))
            .fetchone()[0]
            == 0
        )
        assert manager.transitions(job_id) == ()


@pytest.mark.parametrize("queued", [False, True])
def test_dead_supervisor_releases_queue_or_thread_capacity(tmp_path: Path, queued: bool) -> None:
    """A killed interpreter cannot retain its queued submission or in-process thread."""
    manager = StudioJobManager(
        root=tmp_path,
        allowed_kinds=frozenset({"analysis"}),
        default_timeout_seconds=10.0,
        max_concurrent_jobs=1,
        max_queued_jobs=1,
    )
    release = threading.Event()
    first = None
    if queued:
        first = manager.submit(
            kind="analysis",
            owner="owner",
            request_id=None,
            task=lambda context: (release.wait(10.0) and {}) or {},
        )
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            """
import sys, threading
from pathlib import Path
from sc_neurocore.studio.platform.jobs import StudioJobManager
m=StudioJobManager(root=Path(sys.argv[1]),allowed_kinds=frozenset({'analysis'}),
    default_timeout_seconds=30.0,max_concurrent_jobs=1,max_queued_jobs=1)
m.submit(kind='analysis',owner='child',request_id=None,task=lambda context: threading.Event().wait(30))
threading.Event().wait(30)
""",
            str(tmp_path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            snapshot = manager._admission.snapshot()
            if (snapshot.queued == 1) if queued else (snapshot.running == 1):
                break
            assert child.poll() is None
            time.sleep(0.01)
        assert (
            manager._admission.snapshot().queued
            if queued
            else manager._admission.snapshot().running
        ) == 1
        child.kill()
        child.communicate(timeout=3.0)
        manager.reconcile()
        release.set()
        if first is not None:
            assert manager.wait(first.job_id, 2.0).status == "completed"
        snapshot = manager._admission.snapshot()
        assert snapshot.running == snapshot.queued == 0
        fresh = manager.submit(
            kind="analysis", owner="owner", request_id=None, task=lambda context: {}
        )
        assert manager.wait(fresh.job_id, 2.0).status == "completed"
    finally:
        release.set()
        if child.poll() is None:
            child.kill()
        child.communicate(timeout=3.0)
        if first is not None:
            manager.wait(first.job_id, 2.0)
