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

The race is made deterministic by the ledger connection's own statement trace:
the real job is released and completes between the pre-check read and the
transition write. The ledger, the transition and the manager are the production
ones.
"""

from __future__ import annotations

from pathlib import Path
import os
import threading
import time
from collections.abc import Mapping

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


def peer_cancel_process_task(
    context: StudioJobContext, payload: Mapping[str, object]
) -> dict[str, object]:
    """Expose an actual worker PID and remain alive for the cancellation test."""
    context.write_artifact("worker.pid", str(os.getpid()))
    time.sleep(10.0)
    return {"unexpected_completion": True}


class TestCancellingAJobThatAlreadyStopped:
    def test_a_terminal_job_is_returned_rather_than_transitioned(
        self, manager: StudioJobManager
    ) -> None:
        finished = _finished_job(manager)
        assert finished.status == "completed"

        cancelled = manager.cancel(finished.job_id)

        assert cancelled.status == "completed"

    def test_a_job_that_stops_inside_the_race_window_is_still_a_no_op(
        self, manager: StudioJobManager
    ) -> None:
        """The pre-check sees a live job; the ledger sees a finished one.

        When cancel's connection starts the transition, right after its status
        read, the real job is released and its supervisor commits completion
        on another connection first.
        """
        release, started = threading.Event(), threading.Event()

        def task(context: StudioJobContext) -> dict[str, object]:
            started.set()
            release.wait(10.0)
            return {"done": True}

        submitted = manager.submit(kind="compiler", owner="test", request_id=None, task=task)
        assert started.wait(5.0)
        statements: list[str] = []

        def finish_before_write(statement: str) -> None:
            read = statements and statements[-1].startswith("SELECT * FROM jobs WHERE job_id")
            if read and statement == "BEGIN IMMEDIATE" and not release.is_set():
                release.set()
                assert manager._done_events[submitted.job_id].wait(10.0)
            statements.append(statement)

        connection = manager._ledger.connection()
        connection.set_trace_callback(finish_before_write)
        try:
            cancelled = manager.cancel(submitted.job_id)
        finally:
            connection.set_trace_callback(None)

        assert release.is_set()
        assert cancelled.status == "completed"

    def test_a_refusal_that_is_not_the_race_still_propagates(
        self, manager: StudioJobManager
    ) -> None:
        """Swallowing every refusal would hide a real ledger inconsistency.

        A job whose outcome is ``unknown`` cannot move to ``cancelling``; that
        refusal is not a job that just stopped.
        """
        job_id = "sj_00000000000000aa"
        manager._admission.admit(
            job_id=job_id,
            kind="compiler",
            actor="test",
            workspace="default",
            request_id=None,
            idempotency_key=None,
            experiment_sha256=None,
            admission=None,
            execution_model="thread",
        )
        manager._ledger.transition(job_id, "unknown")

        with pytest.raises(StudioJobRejected, match="cannot move from 'unknown'"):
            manager.cancel(job_id)

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
