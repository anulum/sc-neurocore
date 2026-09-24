# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Live worker capacity custody

"""A terminal record does not release capacity while its real worker lives."""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import pytest

from sc_neurocore.studio.platform.jobs import StudioJobContext, StudioJobManager
from sc_neurocore.studio.platform.jobs_ledger_supervisor import supervisor_identity
from sc_neurocore.studio.platform.jobs_process_state import group_survivors
from tests.studio_seccomp_support import SECCOMP_AVAILABLE, run_child
from sc_neurocore.studio.platform.jobs_reaper import (
    reap_process_group,
)


@pytest.mark.parametrize("token", ["0", "corrupt"])
def test_unknown_supervisor_token_retains_queued_custody(tmp_path: Path, token: str) -> None:
    """Public reconciliation must not release a queue on malformed live identity."""
    manager = StudioJobManager(
        root=tmp_path, allowed_kinds=frozenset({"analysis"}), default_timeout_seconds=3.0
    )
    host, pid, _ = supervisor_identity().split(":", 2)
    identity = f"{host}:{pid}:{token}"
    with manager._ledger.transaction() as connection:
        connection.execute(
            "INSERT INTO admission_reservations(job_id,supervisor,state) VALUES(?,?,'queued')",
            ("sj_0123456789abcdef", identity),
        )
    before = manager.status().admission
    try:
        assert manager.reconcile() == ()
        assert manager.status().admission == before
        row = (
            manager._ledger.connection()
            .execute(
                "SELECT supervisor,state FROM admission_reservations WHERE job_id=?",
                ("sj_0123456789abcdef",),
            )
            .fetchone()
        )
        assert tuple(row) == (identity, "queued")
        assert manager.list_records() == ()
    finally:
        manager._ledger.close()


def task_leaving_descendant(
    context: StudioJobContext, payload: Mapping[str, object]
) -> dict[str, object]:
    """Return a real result while a bounded child still belongs to the worker group."""
    context.write_artifact("group.pid", str(os.getpgrp()))
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(20)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return {"descendant_pid": child.pid}


@pytest.mark.parametrize("use_own_group", [False, True])
def test_reaper_rejects_unowned_group_without_signalling(use_own_group: bool) -> None:
    """Invalid group custody never stops the worker, another group or the caller."""
    processes: list[subprocess.Popen[bytes]] = []
    try:
        for _ in range(2):
            processes.append(
                subprocess.Popen(
                    [sys.executable, "-c", "import time; time.sleep(10)"],
                    start_new_session=True,
                )
            )
        worker, unrelated = processes
        supplied_group = os.getpgrp() if use_own_group else unrelated.pid
        report = reap_process_group(worker, owned_group_id=supplied_group)
        assert not report.reaped and report.outcome == "unreaped"
        assert report.group_id is None
        assert worker.poll() is None and unrelated.poll() is None
    finally:
        for process in processes:
            assert reap_process_group(process, owned_group_id=process.pid).reaped


_CUSTODY_CHILD = (
    "import errno, json, os, signal, sqlite3, sys, threading, time\n"
    "from contextlib import closing\n"
    "from pathlib import Path\n"
    "from sc_neurocore.studio.platform.jobs import StudioJobManager\n"
    "from sc_neurocore.studio.platform.jobs_admission import StudioJobQueueFull\n"
    "from sc_neurocore.studio.platform.jobs_process_state import group_survivors\n"
    "from tests.studio_seccomp_support import Refusal, install_refusals\n"
    "from tests.studio_syscall_support import finish\n"
    "root, trigger = Path(sys.argv[1]), sys.argv[2]\n"
    "def open_manager(timeout):\n"
    "    return StudioJobManager(root=root, allowed_kinds=frozenset({'analysis'}),\n"
    "        default_timeout_seconds=timeout, max_concurrent_jobs=1, max_queued_jobs=0)\n"
    "def queue_full(manager):\n"
    "    try:\n"
    "        manager.submit(kind='analysis', owner='observer', request_id=None,\n"
    "            task=lambda context: {})\n"
    "    except StudioJobQueueFull:\n"
    "        return True\n"
    "    return False\n"
    "def stop_group(group):\n"
    "    for pid in group_survivors(group):\n"
    "        descriptor = os.pidfd_open(pid)\n"
    "        signal.pidfd_send_signal(descriptor, signal.SIGKILL)\n"
    "        os.close(descriptor)\n"
    "    deadline = time.monotonic() + 10.0\n"
    "    while group_survivors(group) and time.monotonic() < deadline:\n"
    "        time.sleep(0.01)\n"
    "    return not group_survivors(group)\n"
    "install_refusals([Refusal('kill', errno.EPERM)])\n"
)


@pytest.mark.parametrize("reap_fails", [False, True])
def test_successful_worker_stops_descendants_before_releasing_capacity(
    tmp_path: Path, reap_fails: bool
) -> None:
    """Completed job custody includes the real descendants, not just the direct worker.

    When the kernel refuses the supervisor's group signals, the descendant
    survives for real and the job keeps its slot.
    """
    if reap_fails:
        if not SECCOMP_AVAILABLE:
            pytest.skip("held system calls need Linux x86_64")
        result = run_child(
            _CUSTODY_CHILD + "manager = open_manager(5.0)\n"
            "record = manager.submit_process_task(kind='analysis', owner='operator',\n"
            "    request_id=None, payload={}, task_path=\n"
            "    'tests.test_studio_jobs_capacity_custody:task_leaving_descendant')\n"
            "outcome = manager.wait(record.job_id, 30.0)\n"
            "group = int((root / record.job_id / 'group.pid').read_text())\n"
            "row = manager._ledger.connection().execute(\n"
            "    'SELECT * FROM job_workers WHERE job_id=?', (record.job_id,)).fetchone()\n"
            "out = {'status': outcome.status, 'error': outcome.error,\n"
            "    'survivors': bool(group_survivors(group)),\n"
            "    'running': manager._admission.snapshot().running,\n"
            "    'registered': row is not None and row['group_id'] == group,\n"
            "    'queue_full': queue_full(manager)}\n"
            "out['cleaned'] = stop_group(group)\n"
            "finish(out)\n",
            arguments=(str(tmp_path), "reap-fails"),
        )
        assert result["status"] == "failed"
        assert isinstance(result["error"], str) and "not reaped" in result["error"]
        assert {key: result[key] for key in ("survivors", "running", "registered")} == {
            "survivors": True,
            "running": 1,
            "registered": True,
        }
        assert (result["queue_full"], result["cleaned"]) == (True, True)
        return
    manager = StudioJobManager(
        root=tmp_path,
        allowed_kinds=frozenset({"analysis"}),
        default_timeout_seconds=5.0,
        max_concurrent_jobs=1,
        max_queued_jobs=0,
    )
    record = manager.submit_process_task(
        kind="analysis",
        owner="operator",
        request_id=None,
        task_path="tests.test_studio_jobs_capacity_custody:task_leaving_descendant",
        payload={},
    )
    marker = tmp_path / record.job_id / "group.pid"
    try:
        outcome = manager.wait(record.job_id, 8.0)
        assert marker.exists()
        group_id = int(marker.read_text())
        identity = (
            manager._ledger.connection()
            .execute("SELECT * FROM job_workers WHERE job_id=?", (record.job_id,))
            .fetchone()
        )
        assert identity is not None
        assert identity["group_id"] == group_id
        assert identity["supervisor"] == manager._ledger.supervisor
        assert identity["worker_identity"].split(":")[1] == str(group_id)
        assert identity["boot_id"] == Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        assert outcome.status == "completed"
        assert outcome.result is not None and "descendant_pid" in outcome.result
        assert not group_survivors(group_id), "Completed job left a live descendant"
        assert manager._admission.snapshot().running == 0
    finally:
        manager.wait(record.job_id, 2.0)


@pytest.mark.skipif(not SECCOMP_AVAILABLE, reason="held system calls need Linux x86_64")
@pytest.mark.parametrize("trigger", ["cancel", "timeout", "observation-failure"])
def test_unreaped_process_retains_capacity(tmp_path: Path, trigger: str) -> None:
    """A real process left alive by a refused reap still occupies the shared slot.

    The kernel refuses the supervisor's signals (``kill``). For
    ``observation-failure`` another connection really corrupts the job's
    stored artifacts, so the supervisor's next observation fails.
    """
    result = run_child(
        _CUSTODY_CHILD + "manager = open_manager(3.0 if trigger == 'timeout' else 30.0)\n"
        "record = manager.submit_process_task(kind='analysis', owner='operator',\n"
        "    request_id=None, payload={'seconds': 120},\n"
        "    task_path='tests.studio_job_tasks:process_sleep_task')\n"
        "registered = 'SELECT group_id FROM job_workers WHERE job_id=?'\n"
        "deadline = time.monotonic() + 15.0\n"
        "row = None\n"
        "while row is None and time.monotonic() < deadline:\n"
        "    row = manager._ledger.connection().execute(registered, (record.job_id,)).fetchone()\n"
        "    time.sleep(0.01)\n"
        "worker = int(row[0])\n"
        "if trigger == 'cancel':\n"
        "    manager.cancel(record.job_id)\n"
        "if trigger == 'observation-failure':\n"
        "    with closing(sqlite3.connect(manager.ledger_path, isolation_level=None)) as other:\n"
        "        other.execute(\"UPDATE jobs SET artifacts='[{}]' WHERE job_id=?\", (record.job_id,))\n"
        "# The corrupt row is readable again only once the supervisor settled it.\n"
        "done = manager._done_events[record.job_id].wait(20.0)\n"
        "outcome = manager.record(record.job_id)\n"
        "out = {'status': outcome.status, 'error': outcome.error, 'done': done,\n"
        "    'alive': os.path.exists(f'/proc/{worker}') and worker in group_survivors(worker),\n"
        "    'running': manager._admission.snapshot().running}\n"
        "observer = open_manager(3.0)\n"
        "out['observed'] = observer._admission.snapshot().running\n"
        "out['observer_full'] = queue_full(observer)\n"
        "manager._admission.release(job_id=record.job_id)\n"
        "out['after_release'] = manager._admission.snapshot().running\n"
        "out['manager_full'] = queue_full(manager)\n"
        "out['cleaned'] = stop_group(worker)\n"
        "finish(out)\n",
        arguments=(str(tmp_path), trigger),
    )
    status = {"cancel": "cancelled", "timeout": "timed_out", "observation-failure": "failed"}[
        trigger
    ]
    assert result["status"] == status
    assert isinstance(result["error"], str) and "not reaped" in result["error"]
    if trigger == "observation-failure":
        assert "observation failed" in result["error"]
    assert {key: value for key, value in result.items() if key not in ("status", "error")} == {
        "done": True,
        "alive": True,
        "running": 1,
        "observed": 1,
        "observer_full": True,
        "after_release": 1,
        "manager_full": True,
        "cleaned": True,
    }
