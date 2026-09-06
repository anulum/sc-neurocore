# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio job cancellation against a job that already stopped

"""Cancelling a job that has already stopped, including in the race window.

``cancel`` reads the job's status and then writes the transition, and the read
is outside the ledger transaction. Under load a run reaches a terminal state in
that gap, the ledger refuses ``cancelled -> cancelling``, and the refusal
reached the operator as a server error for pressing Stop on a run that had just
finished. Observed in a loaded slice run as
``Studio job sj_94b53edeb87d968a cannot move from 'cancelled' to 'cancelling'``.

The race is made deterministic here by holding the pre-check's answer stale for
exactly one call, which is the only way to drive that window on purpose; the
ledger, the transition and the manager are the production ones.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from sc_neurocore.studio.platform.jobs import (
    StudioJobContext,
    StudioJobManager,
    StudioJobRecord,
    StudioJobRejected,
)


@pytest.fixture
def manager(tmp_path: Path) -> StudioJobManager:
    """A job manager whose ledger lives under the test's own directory."""
    return StudioJobManager(
        root=tmp_path / "jobs",
        allowed_kinds=frozenset({"compiler"}),
        default_timeout_seconds=5.0,
    )


def _finished_job(manager: StudioJobManager) -> StudioJobRecord:
    """Submit and complete one job, returning its terminal record."""

    def task(context: StudioJobContext) -> dict[str, object]:
        return {"done": True}

    submitted = manager.submit(kind="compiler", owner="test", request_id=None, task=task)
    return manager.wait(submitted.job_id, timeout_seconds=5.0)


class TestCancellingAJobThatAlreadyStopped:
    def test_a_terminal_job_is_returned_rather_than_transitioned(
        self, manager: StudioJobManager
    ) -> None:
        finished = _finished_job(manager)
        assert finished.status == "completed"

        cancelled = manager.cancel(finished.job_id)

        assert cancelled.status == "completed"

    def test_a_job_that_stops_inside_the_race_window_is_still_a_no_op(
        self, manager: StudioJobManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The pre-check sees a live job; the ledger sees a finished one."""
        finished = _finished_job(manager)
        ledger = manager._ledger
        real_record = ledger.record
        stale = [True]

        def racing_record(job_id: str) -> StudioJobRecord:
            record = real_record(job_id)
            if stale[0]:
                stale[0] = False
                return replace(record, status="running")
            return record

        monkeypatch.setattr(ledger, "record", racing_record)

        cancelled = manager.cancel(finished.job_id)

        assert cancelled.status == "completed"
        assert stale[0] is False

    def test_a_refusal_that_is_not_the_race_still_propagates(
        self, manager: StudioJobManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Swallowing every refusal would hide a real ledger inconsistency."""
        finished = _finished_job(manager)
        ledger = manager._ledger
        real_record = ledger.record

        def always_running(job_id: str) -> StudioJobRecord:
            return replace(real_record(job_id), status="running")

        monkeypatch.setattr(ledger, "record", always_running)

        with pytest.raises(StudioJobRejected, match="cannot move from 'completed'"):
            manager.cancel(finished.job_id)

    def test_cancelling_twice_is_a_no_op_the_second_time(self, manager: StudioJobManager) -> None:
        def task(context: StudioJobContext) -> dict[str, object]:
            return {"done": True}

        submitted = manager.submit(kind="compiler", owner="test", request_id=None, task=task)
        manager.wait(submitted.job_id, timeout_seconds=5.0)

        first = manager.cancel(submitted.job_id)
        second = manager.cancel(submitted.job_id)

        assert first.status == second.status == "completed"

    def test_an_unknown_job_is_still_a_key_error(self, manager: StudioJobManager) -> None:
        with pytest.raises(KeyError):
            manager.cancel("sj_0000000000000000")
