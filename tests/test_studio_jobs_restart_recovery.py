# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio job custody across restarts

"""Custody has to survive the process, so these tests kill real processes.

Each case uses a separate interpreter over one shared job root. The point is
never that a status string changed — it is that a restarted Studio tells the
truth about work it cannot account for, instead of losing it or claiming it
finished.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from sc_neurocore.studio.platform.jobs import StudioJobManager
from sc_neurocore.studio.platform.jobs_context import StudioJobContext
from sc_neurocore.studio.platform.jobs_models import StudioJobRejected

REPO_ROOT = Path(__file__).resolve().parents[1]

_CHILD = """
import json, sys, time
from pathlib import Path
from sc_neurocore.studio.platform.jobs import StudioJobManager

root = Path(sys.argv[1])
mode = sys.argv[2]
manager = StudioJobManager(
    root=root, allowed_kinds=frozenset({"analysis"}), default_timeout_seconds=120.0
)

if mode == "complete":
    record = manager.submit(
        kind="analysis",
        owner="alice",
        request_id="req-1",
        task=lambda context: {"answer": 42},
        idempotency_key="key-1",
        experiment_sha256="a" * 64,
    )
    done = manager.wait(record.job_id, 60.0)
    print(json.dumps({"job_id": done.job_id, "status": done.status}))
elif mode == "abandon":
    def forever(context):
        # The artifact reaches disk; its manifest never reaches the ledger,
        # because this process is killed before the terminal transition.
        context.write_artifact("partial.bin", b"written before the crash")
        while True:
            time.sleep(0.05)

    record = manager.submit(
        kind="analysis", owner="alice", request_id="req-2", task=forever
    )
    while manager.record(record.job_id).status != "running":
        time.sleep(0.02)
    (root / "started.json").write_text(json.dumps({"job_id": record.job_id}))
    time.sleep(3600)
"""


def _run_child(
    root: Path, mode: str, *, wait: bool = True
) -> subprocess.CompletedProcess[str] | subprocess.Popen[str]:
    """Run one child interpreter over the shared job root."""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(REPO_ROOT / "src")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    command = [sys.executable, "-c", _CHILD, str(root), mode]
    if wait:
        return subprocess.run(  # noqa: S603 - fixed argv, no shell
            command, capture_output=True, text=True, env=environment, timeout=600, check=False
        )
    return subprocess.Popen(  # noqa: S603 - fixed argv, no shell
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
        start_new_session=True,
    )


def _manager(root: Path, **kwargs: object) -> StudioJobManager:
    """Open a manager over the shared root, reconciling on construction."""
    return StudioJobManager(
        root=root,
        allowed_kinds=frozenset({"analysis"}),
        default_timeout_seconds=30.0,
        **kwargs,  # type: ignore[arg-type]
    )


class TestTwoProcessesShareOneRoot:
    @pytest.mark.parametrize("observer", ["same-manager", "second-manager"])
    @pytest.mark.parametrize("expired", [False, True], ids=["fresh-lease", "expired-lease"])
    def test_recovery_preserves_a_job_running_in_this_process(
        self, tmp_path: Path, observer: str, expired: bool
    ) -> None:
        """Recovery must not invent supervisor death while a real task is running."""
        root = tmp_path / "jobs"
        moment = [datetime.now(timezone.utc)]
        manager = _manager(root, clock=lambda: moment[0])
        started = threading.Event()
        release = threading.Event()

        def task(context: StudioJobContext) -> dict[str, object]:
            started.set()
            if not release.wait(10.0):
                raise TimeoutError("test did not release the job")
            context.write_artifact("result.txt", b"completed after recovery")
            return {"answer": 42}

        record = manager.submit(kind="analysis", owner="alice", request_id=None, task=task)
        try:
            assert started.wait(5.0)
            if expired:
                moment[0] += timedelta(seconds=120)
            if observer == "same-manager":
                decisions = manager.reconcile()
                reader = manager
            else:
                reader = _manager(root, clock=lambda: moment[0])
                decisions = reader.last_reconciliation
            assert reader.record(record.job_id).status == "running"
            assert [item.status for item in decisions] == ["running"]
            assert reader.status().recovery[0]["status"] == "running"
        finally:
            release.set()
            settled = manager.wait(record.job_id, 5.0)
        assert settled.status == "completed"
        assert settled.result == {"answer": 42}
        assert (root / record.job_id / "result.txt").read_bytes() == b"completed after recovery"
        assert [item["to_status"] for item in reader.transitions(record.job_id)] == [
            "pending",
            "running",
            "completed",
        ]

    def test_a_second_process_reads_a_job_the_first_completed(self, tmp_path: Path) -> None:
        root = tmp_path / "jobs"
        completed = _run_child(root, "complete")
        assert isinstance(completed, subprocess.CompletedProcess)
        assert completed.returncode == 0, completed.stderr
        reported = json.loads(completed.stdout)
        assert reported["status"] == "completed"

        # A different interpreter, after the first one exited.
        manager = _manager(root)
        record = manager.wait(reported["job_id"], timeout_seconds=0.1)

        assert record.status == "completed"
        assert record.result == {"answer": 42}
        assert record.experiment_sha256 == "a" * 64
        assert [entry["to_status"] for entry in manager.transitions(record.job_id)] == [
            "pending",
            "running",
            "completed",
        ]

    def test_a_restarted_process_does_not_rerun_an_admitted_request(self, tmp_path: Path) -> None:
        root = tmp_path / "jobs"
        first = _run_child(root, "complete")
        assert isinstance(first, subprocess.CompletedProcess)
        assert first.returncode == 0, first.stderr
        original = json.loads(first.stdout)["job_id"]

        manager = _manager(root)
        resubmitted = manager.submit(
            kind="analysis",
            owner="alice",
            request_id="req-1",
            task=lambda context: {"answer": 0},
            idempotency_key="key-1",
        )

        assert resubmitted.job_id == original
        assert resubmitted.result == {"answer": 42}
        assert len(manager.list_records()) == 1

    @pytest.mark.parametrize("observer", ["same-manager", "second-manager"])
    def test_wait_deadline_returns_live_record_without_cancelling_work(
        self, tmp_path: Path, observer: str
    ) -> None:
        """A wait timeout is observation, not a job timeout or cancellation request."""
        root = tmp_path / "jobs"
        manager = _manager(root)
        started = threading.Event()
        release = threading.Event()

        def task(context: StudioJobContext) -> dict[str, object]:
            started.set()
            if not release.wait(5.0):
                raise TimeoutError("test did not release the job")
            context.check_cancelled()
            return {"answer": 42}

        submitted = manager.submit(kind="analysis", owner="alice", request_id=None, task=task)
        try:
            assert started.wait(2.0)
            reader = manager if observer == "same-manager" else _manager(root)
            observed = reader.wait(submitted.job_id, timeout_seconds=0.02)
            assert observed.status == "running"
            assert observed.finished_at_utc is None
            assert observed.result is None
            assert reader.wait(submitted.job_id, timeout_seconds=0.0) == observed
            assert reader.wait(submitted.job_id, timeout_seconds=-1.0) == observed
            timer = threading.Timer(0.05, release.set)
            timer.start()
            try:
                assert reader.wait(submitted.job_id, timeout_seconds=5.0).status == "completed"
            finally:
                timer.join(timeout=1.0)
        finally:
            release.set()
            settled = manager.wait(submitted.job_id, timeout_seconds=5.0)
        assert settled.status == "completed"
        assert reader.wait(submitted.job_id) == settled
        assert reader.wait(submitted.job_id, timeout_seconds=0.0) == settled
        assert [row["to_status"] for row in reader.transitions(submitted.job_id)] == [
            "pending",
            "running",
            "completed",
        ]

    @pytest.mark.parametrize("timeout", [float("nan"), float("inf"), float("-inf")])
    def test_wait_rejects_nonfinite_deadlines(self, tmp_path: Path, timeout: float) -> None:
        """Invalid observation deadlines fail explicitly rather than wait forever."""
        manager = _manager(tmp_path / "jobs")
        with pytest.raises(ValueError, match="finite"):
            manager.wait("sj_absent", timeout_seconds=timeout)

    def test_wait_refuses_an_absent_record(self, tmp_path: Path) -> None:
        """Missing durable jobs still raise KeyError instead of appearing complete."""
        manager = _manager(tmp_path / "jobs")
        with pytest.raises(KeyError):
            manager.wait("sj_absent", timeout_seconds=0.0)


class TestSupervisorDeath:
    def test_a_killed_supervisor_leaves_an_interrupted_job_not_a_lost_one(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / "jobs"
        child = _run_child(root, "abandon", wait=False)
        assert isinstance(child, subprocess.Popen)
        started = root / "started.json"
        try:
            deadline = time.monotonic() + 90
            while not started.is_file() and time.monotonic() < deadline:
                if child.poll() is not None:  # pragma: no cover - child failure
                    assert child.stderr is not None
                    raise AssertionError(f"the child exited early: {child.stderr.read()}")
                time.sleep(0.05)
            assert started.is_file(), "the child never started its job"
            job_id = json.loads(started.read_text())["job_id"]
        finally:
            os.killpg(os.getpgid(child.pid), signal.SIGKILL)
            child.wait(timeout=60)

        # Restarting reconciles: the job is accounted for, not silently gone
        # and not silently finished.
        manager = _manager(root)
        record = manager.record(job_id)

        assert record.status == "interrupted"
        assert record.result is None
        assert record.error is not None and "did not finish" in record.error
        assert record.finished_at_utc is not None
        decisions = {decision.job_id: decision for decision in manager.last_reconciliation}
        assert decisions[job_id].previous_status == "running"
        assert decisions[job_id].status == "interrupted"

        # The artifact the job wrote before dying is still on disk. Recovery
        # neither deletes that evidence nor promotes it into a manifest the
        # job never committed.
        assert (root / job_id / "partial.bin").read_bytes() == b"written before the crash"
        assert record.artifacts == ()

        # And the interrupted outcome is final: a later claim of success is
        # refused rather than overwriting what happened.
        with pytest.raises(StudioJobRejected, match="cannot move"):
            manager._ledger.transition(job_id, "completed", result={"answer": 42})
