# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio worker reaping and honest cancellation

"""Stopping a job has to stop the work, or say that it did not.

These cases use real processes that resist being stopped: one that ignores
SIGTERM, one that spawns a child, and one that never checks for cancellation at
all. The assertion is never that a status string changed — it is that nothing
is still running behind a record that says the job ended, or that the record
says so.
"""

from __future__ import annotations

import errno
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from sc_neurocore.studio.platform.jobs import StudioJobContext, StudioJobManager
from sc_neurocore.studio.platform import jobs_process_state
from sc_neurocore.studio.platform.jobs_reaper import (
    ReapReport,
    process_group_of,
    reap_process_group,
)
from tests.studio_seccomp_support import SECCOMP_AVAILABLE, Refusal, run_refused


def _spawn(code: str, *, new_session: bool = True) -> subprocess.Popen[bytes]:
    return subprocess.Popen(  # noqa: S603 - fixed argv, no shell
        [sys.executable, "-c", code], start_new_session=new_session
    )


def _wait_for(path: Path, *, timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    while not path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert path.exists(), f"{path.name} never appeared"


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


class TestReaper:
    def test_a_missing_proc_stat_is_not_treated_as_a_live_worker(self) -> None:
        assert jobs_process_state.process_exited(2**31) is True

    @pytest.mark.skipif(not SECCOMP_AVAILABLE, reason="seccomp filters need Linux x86_64")
    def test_an_unreaped_group_reports_its_live_worker(self) -> None:
        """Signals the kernel refuses leave the worker live and reported, not assumed gone."""
        result = run_refused(
            "import os, signal, subprocess\n"
            "from sc_neurocore.studio.platform.jobs_reaper import reap_process_group\n"
            "worker = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)'],\n"
            "    start_new_session=True)\n"
            "install_refusals(REFUSALS)\n"
            "report = reap_process_group(worker, terminate_grace_seconds=0.2,\n"
            "    kill_grace_seconds=0.2)\n"
            "alive = worker.poll() is None\n"
            "descriptor = os.pidfd_open(worker.pid)\n"
            "signal.pidfd_send_signal(descriptor, signal.SIGKILL)\n"
            "worker.wait(timeout=10)\n"
            "os.close(descriptor)\n"
            "print(json.dumps({'outcome': report.outcome, 'reaped': report.reaped,\n"
            "    'survivor': worker.pid in report.survivors, 'alive': alive}))\n",
            [Refusal("kill", errno.EPERM)],
        )
        assert result == {"outcome": "unreaped", "reaped": False, "survivor": True, "alive": True}

    def test_a_finished_worker_is_reported_as_exited(self) -> None:
        process = _spawn("pass")
        process.wait(timeout=60)

        report = reap_process_group(process)

        assert report.outcome == "exited"
        assert report.reaped is True

    def test_a_worker_that_ignores_sigterm_is_killed_and_reaped(self, tmp_path: Path) -> None:
        ready = tmp_path / "ready"
        process = _spawn(
            "import signal, time\n"
            "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
            f"open({str(ready)!r}, 'w').write('1')\n"
            "time.sleep(300)\n"
        )
        _wait_for(ready)

        report = reap_process_group(process, terminate_grace_seconds=0.3)

        assert report.outcome == "killed"
        assert report.reaped is True
        assert report.survivors == ()
        assert _alive(process.pid) is False

    def test_the_worker_s_own_children_are_reaped_with_it(self, tmp_path: Path) -> None:
        marker = tmp_path / "child.pid"
        process = _spawn(
            "import subprocess, sys, time\n"
            "subprocess.Popen([sys.executable, '-c', "
            f"\"import os, time; open({str(marker)!r}, 'w').write(str(os.getpid())); time.sleep(300)\"])\n"
            "time.sleep(300)\n"
        )
        _wait_for(marker)
        # The file exists before the child finished writing its PID into it.
        deadline = time.monotonic() + 10.0
        while not marker.read_text() and time.monotonic() < deadline:
            time.sleep(0.01)
        child_pid = int(marker.read_text())
        assert _alive(child_pid) is True

        report = reap_process_group(process)

        time.sleep(0.2)
        assert report.reaped is True
        # The whole group went, not only the process the supervisor could see.
        assert _alive(child_pid) is False

    @pytest.mark.skipif(not SECCOMP_AVAILABLE, reason="seccomp filters need Linux x86_64")
    @pytest.mark.parametrize("refused", [signal.SIGTERM, signal.SIGKILL])
    def test_a_direct_child_the_kernel_will_not_stop_is_reported_unreaped(
        self, refused: signal.Signals
    ) -> None:
        """A worker outside its own group that survives both signals is never 'reaped'.

        The worker ignores SIGTERM; the kernel refuses the ``refused`` signal.
        """
        result = run_refused(
            "import os, signal, subprocess\n"
            "from sc_neurocore.studio.platform.jobs_reaper import reap_process_group\n"
            "worker = subprocess.Popen([sys.executable, '-c', 'import signal, time; '\n"
            "    'signal.signal(signal.SIGTERM, signal.SIG_IGN); print(1, flush=True); '\n"
            "    'time.sleep(60)'], stdout=subprocess.PIPE)\n"
            "worker.stdout.readline()\n"
            "install_refusals(REFUSALS)\n"
            "report = reap_process_group(worker, terminate_grace_seconds=0.2,\n"
            "    kill_grace_seconds=0.2)\n"
            "alive = worker.poll() is None\n"
            "descriptor = os.pidfd_open(worker.pid)\n"
            "signal.pidfd_send_signal(descriptor, signal.SIGKILL)\n"
            "worker.wait(timeout=10)\n"
            "print(json.dumps({'outcome': report.outcome, 'reaped': report.reaped,\n"
            "    'survivor': worker.pid in report.survivors, 'alive': alive}))\n",
            [Refusal("kill", errno.EPERM, int(refused), argument_index=1)],
        )
        assert result == {"outcome": "unreaped", "reaped": False, "survivor": True, "alive": True}

    def test_a_worker_without_its_own_group_is_still_stopped(self) -> None:
        # A worker sharing the supervisor's group cannot be signalled as a
        # group without killing the Studio; only the direct child is stopped,
        # and the reap says so by reporting no group.
        process = _spawn("import time; time.sleep(300)", new_session=False)
        assert process_group_of(process) == os.getpgrp()

        report = reap_process_group(process, terminate_grace_seconds=0.3)

        assert report.group_id is None
        assert _alive(process.pid) is False

    def test_a_reap_report_is_path_free(self) -> None:
        report = ReapReport(outcome="killed", group_id=1234, returncode=-9, duration_seconds=0.5)
        payload = report.to_public_dict()
        assert payload == {
            "duration_seconds": 0.5,
            "group_id": 1234,
            "outcome": "killed",
            "reaped": True,
            "returncode": -9,
            "survivor_count": 0,
        }


class TestUncooperativeThreadJob:
    def test_a_thread_that_never_checks_cancellation_is_reported_not_hidden(
        self, tmp_path: Path
    ) -> None:
        manager = StudioJobManager(
            root=tmp_path / "jobs",
            allowed_kinds=frozenset({"analysis"}),
            default_timeout_seconds=0.3,
        )
        stop = threading.Event()

        def uncooperative(context: StudioJobContext) -> dict[str, object]:
            del context
            while not stop.is_set():
                sum(index * index for index in range(20_000))
            return {"done": True}

        try:
            record = manager.submit(
                kind="analysis", owner="operator", request_id="req-1", task=uncooperative
            )
            completed = manager.wait(record.job_id, timeout_seconds=30.0)

            assert completed.status == "timed_out"
            # The record does not claim the work stopped, because it did not.
            assert "did not stop" in (completed.error or "")
            assert "process job" in (completed.error or "")
            assert manager.unreaped_workers == (record.job_id,)
            assert manager.status().to_public_dict()["unreaped_workers"] == [record.job_id]
        finally:
            stop.set()
            time.sleep(0.2)

    def test_a_cooperative_thread_stops_and_is_not_reported(self, tmp_path: Path) -> None:
        manager = StudioJobManager(
            root=tmp_path / "jobs",
            allowed_kinds=frozenset({"analysis"}),
            default_timeout_seconds=0.3,
        )

        def cooperative(context: StudioJobContext) -> dict[str, object]:
            while not context.cancelled:
                time.sleep(0.01)
            return {}

        record = manager.submit(
            kind="analysis", owner="operator", request_id="req-1", task=cooperative
        )
        completed = manager.wait(record.job_id, timeout_seconds=30.0)

        assert completed.status == "timed_out"
        assert completed.error == "Studio job exceeded its timeout."
        assert manager.unreaped_workers == ()


class TestTerminalSeal:
    def test_a_timed_out_process_job_cannot_be_completed_afterwards(self, tmp_path: Path) -> None:
        from sc_neurocore.studio.platform.jobs_models import StudioJobRejected

        manager = StudioJobManager(
            root=tmp_path / "jobs",
            allowed_kinds=frozenset({"compiler"}),
            default_timeout_seconds=0.05,
        )
        record = manager.submit_process_task(
            kind="compiler",
            owner="operator",
            request_id="req-1",
            task_path="tests.studio_job_tasks:process_sleep_task",
            payload={"seconds": 3},
        )
        completed = manager.wait(record.job_id, timeout_seconds=30.0)
        assert completed.status == "timed_out"

        # A worker that outlived its deadline cannot post a result afterwards.
        with pytest.raises(StudioJobRejected, match="cannot move"):
            manager._ledger.transition(record.job_id, "completed", result={"late": True})
        assert manager.record(record.job_id).result is None


class TestModelScanCancellation:
    def test_a_cancelled_scan_stops_between_models(self) -> None:
        from sc_neurocore.studio.model_scan import scan_all_models
        from sc_neurocore.studio.platform.jobs_models import StudioJobCancelled

        with pytest.raises(StudioJobCancelled, match="cancelled after 0 of"):
            scan_all_models(current=1.0, duration=1.0, should_stop=lambda: True)

    def test_an_uncancelled_scan_still_covers_the_catalogue(self) -> None:
        from sc_neurocore.studio.model_scan import scan_all_models

        payload = scan_all_models(current=10.0, duration=20.0, should_stop=lambda: False)

        models = payload["models"]
        assert isinstance(models, list) and len(models) > 100
