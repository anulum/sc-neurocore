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
import sqlite3
import subprocess
import time
from pathlib import Path

import pytest

from sc_neurocore.studio.platform.jobs_models import StudioJobRejected

REPO_ROOT = Path(__file__).resolve().parents[1]

from tests.test_studio_jobs_restart_recovery import _manager, _run_child


class TestSupervisorDeath:
    @pytest.mark.parametrize("purge", [False, True], ids=["completed", "purged"])
    def test_recovery_retains_a_job_settled_after_its_snapshot(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, purge: bool
    ) -> None:
        """A real child completion after snapshot must not break recovery or lose custody."""
        root = tmp_path / "jobs"
        child = _run_child(root, "complete_on_signal", wait=False)
        assert isinstance(child, subprocess.Popen)
        try:
            marker = root / "started.json"
            deadline = time.monotonic() + 10.0
            while not marker.exists() and time.monotonic() < deadline:
                assert child.poll() is None
                time.sleep(0.01)
            assert marker.exists()
            job_id = json.loads(marker.read_text())["job_id"]
            manager = _manager(root, reconcile=False)
            live_rows = manager._ledger.live_rows

            def snapshot_then_complete() -> tuple[sqlite3.Row, ...]:
                rows = live_rows()
                assert len(rows) == 1 and rows[0]["status"] == "running"
                (root / "finish").touch()
                stdout, stderr = child.communicate(timeout=10.0)
                assert child.returncode == 0, stderr
                assert json.loads(stdout)["status"] == "completed"
                if purge:
                    manager.purge_terminal_record(job_id)
                return rows

            monkeypatch.setattr(manager._ledger, "live_rows", snapshot_then_complete)
            decisions = manager.reconcile()
            if purge:
                assert decisions == ()
                assert manager.list_records() == ()
                assert not (root / job_id).exists()
            else:
                assert decisions[0].status == "completed"
                assert manager.record(job_id).result == {"answer": 42}
                assert (
                    manager.read_artifact(job_id, "result.bin").payload
                    == b"completed during recovery"
                )
                assert [item["to_status"] for item in manager.transitions(job_id)] == [
                    "pending",
                    "running",
                    "completed",
                ]
        finally:
            if child.poll() is None:
                os.killpg(os.getpgid(child.pid), signal.SIGKILL)
                child.wait(timeout=5.0)

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
