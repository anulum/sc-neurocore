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
import time
from pathlib import Path

import pytest

from sc_neurocore.studio.platform.jobs import StudioJobManager
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
    def test_a_second_process_reads_a_job_the_first_completed(self, tmp_path: Path) -> None:
        root = tmp_path / "jobs"
        completed = _run_child(root, "complete")
        assert completed.returncode == 0, completed.stderr  # type: ignore[union-attr]
        reported = json.loads(completed.stdout)  # type: ignore[union-attr]
        assert reported["status"] == "completed"

        # A different interpreter, after the first one exited.
        manager = _manager(root)
        record = manager.record(reported["job_id"])

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
        assert first.returncode == 0, first.stderr  # type: ignore[union-attr]
        original = json.loads(first.stdout)["job_id"]  # type: ignore[union-attr]

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


class TestSupervisorDeath:
    def test_a_killed_supervisor_leaves_an_interrupted_job_not_a_lost_one(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / "jobs"
        child = _run_child(root, "abandon", wait=False)
        started = root / "started.json"
        try:
            deadline = time.monotonic() + 90
            while not started.is_file() and time.monotonic() < deadline:
                if child.poll() is not None:  # pragma: no cover - child failure
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
