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
import threading
import os
import time
import sqlite3
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
    def test_repeat_stop_delivers_the_local_event(self, manager: StudioJobManager) -> None:
        """An already-durable request does not skip delivery to the local worker."""
        started, release = threading.Event(), threading.Event()

        def task(context: StudioJobContext) -> dict[str, object]:
            started.set()
            release.wait(3.0)
            context.check_cancelled()
            return {}

        submitted = manager.submit(kind="compiler", owner="test", request_id=None, task=task)
        try:
            assert started.wait(2.0)
            manager._ledger.transition(submitted.job_id, "cancelling")
            assert manager.cancel(submitted.job_id).status == "cancelling"
            assert manager._cancel_events[submitted.job_id].is_set()
        finally:
            release.set()
            assert manager.wait(submitted.job_id, 3.0).status == "cancelled"

    def test_read_failure_reports_a_thread_that_does_not_stop(
        self, manager: StudioJobManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Control-store failure cannot claim that an uncooperative task was killed."""
        started, release, exited = threading.Event(), threading.Event(), threading.Event()

        def task(context: StudioJobContext) -> dict[str, object]:
            started.set()
            try:
                release.wait(5.0)
                return {"late": True}
            finally:
                exited.set()

        submitted = manager.submit(kind="compiler", owner="test", request_id=None, task=task)
        try:
            assert started.wait(2.0)
            real_record = manager._ledger.record

            def failing_read(
                job_id: str, *, actor: str | None = None, workspace: str | None = None
            ) -> StudioJobRecord:
                if threading.current_thread() is not threading.main_thread():
                    raise sqlite3.OperationalError("injected observation failure")
                return real_record(job_id, actor=actor, workspace=workspace)

            monkeypatch.setattr(manager._ledger, "record", failing_read)
            outcome = manager.wait(submitted.job_id, 3.0)
            assert outcome.status == "failed"
            assert outcome.error is not None and "Worker stopped: False" in outcome.error
            assert outcome.result is None
            assert not exited.is_set()
            release.set()
            assert exited.wait(2.0)
            assert manager.record(submitted.job_id) == outcome
        finally:
            release.set()
            assert exited.wait(2.0)

    @pytest.mark.parametrize(
        "read_failure,write_failure", [(False, False), (True, False), (True, True)]
    )
    def test_peer_request_reaps_the_owning_process(
        self,
        manager: StudioJobManager,
        monkeypatch: pytest.MonkeyPatch,
        read_failure: bool,
        write_failure: bool,
    ) -> None:
        """Stop from another manager reaps an actual already-started child."""
        peer = StudioJobManager(
            root=manager._root,
            allowed_kinds=frozenset({"compiler"}),
            default_timeout_seconds=15.0,
        )
        submitted = manager.submit_process_task(
            kind="compiler",
            owner="test",
            request_id=None,
            task_path="tests.test_studio_jobs_cancel_race:peer_cancel_process_task",
            payload={},
            timeout_seconds=15.0,
        )
        try:
            marker = manager._root / submitted.job_id / "worker.pid"
            deadline = time.monotonic() + 5.0
            while not marker.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            assert marker.exists()
            pid = int(marker.read_text())
            os.kill(pid, 0)
            failed = threading.Event()
            errors: list[BaseException] = []
            if write_failure:

                def capture_error(args: threading.ExceptHookArgs) -> None:
                    if args.exc_value is not None:
                        errors.append(args.exc_value)
                    failed.set()

                def refuse_update(*args: object, **kwargs: object) -> None:
                    raise sqlite3.OperationalError("injected terminal write failure")

                monkeypatch.setattr(threading, "excepthook", capture_error)
                monkeypatch.setattr(manager, "_update", refuse_update)
            if read_failure:
                real_record = manager._ledger.record

                def failing_read(
                    job_id: str, *, actor: str | None = None, workspace: str | None = None
                ) -> StudioJobRecord:
                    if threading.current_thread() is not threading.main_thread():
                        raise sqlite3.OperationalError("injected observation failure")
                    return real_record(job_id, actor=actor, workspace=workspace)

                monkeypatch.setattr(manager._ledger, "record", failing_read)
            else:
                assert peer.cancel(submitted.job_id).status in {"cancelling", "cancelled"}
            if write_failure:
                assert failed.wait(3.0)
                assert len(errors) == 1
                assert isinstance(errors[0], sqlite3.OperationalError)
                assert str(errors[0]) == "injected terminal write failure"
                assert manager._done_events[submitted.job_id].is_set()
                retained = peer.wait(submitted.job_id, 0.02)
                assert retained.status == "running" and retained.result is None
                with pytest.raises(ProcessLookupError):
                    os.kill(pid, 0)
                return
            outcome = peer.wait(submitted.job_id, 3.0)
            assert outcome.status == ("failed" if read_failure else "cancelled")
            assert outcome.result is None
            if read_failure:
                assert outcome.error is not None and "Worker reaped" in outcome.error
            else:
                assert outcome.error is None
            with pytest.raises(ProcessLookupError):
                os.kill(pid, 0)
        finally:
            manager.cancel(submitted.job_id)
            if not write_failure:
                manager.wait(submitted.job_id, 8.0)

    @pytest.mark.parametrize("read_failure", [False, True])
    def test_peer_request_reaches_the_owning_thread(
        self, manager: StudioJobManager, monkeypatch: pytest.MonkeyPatch, read_failure: bool
    ) -> None:
        """A second manager's durable Stop request reaches the real task context."""
        started = threading.Event()
        release = threading.Event()

        def task(context: StudioJobContext) -> dict[str, object]:
            started.set()
            while not release.wait(0.01):
                context.check_cancelled()
            context.check_cancelled()
            return {"done": True}

        peer = StudioJobManager(
            root=manager._root,
            allowed_kinds=frozenset({"compiler"}),
            default_timeout_seconds=5.0,
        )
        submitted = manager.submit(kind="compiler", owner="test", request_id=None, task=task)
        try:
            assert started.wait(2.0)
            if read_failure:
                real_record = manager._ledger.record

                def failing_read(
                    job_id: str, *, actor: str | None = None, workspace: str | None = None
                ) -> StudioJobRecord:
                    if threading.current_thread() is not threading.main_thread():
                        raise sqlite3.OperationalError("injected observation failure")
                    return real_record(job_id, actor=actor, workspace=workspace)

                monkeypatch.setattr(manager._ledger, "record", failing_read)
            else:
                requested = peer.cancel(submitted.job_id)
                assert requested.status in {"cancelling", "cancelled"}
            outcome = peer.wait(submitted.job_id, 2.0)
            assert outcome.status == ("failed" if read_failure else "cancelled")
            if read_failure:
                assert outcome.error is not None and "Worker stopped: True" in outcome.error
            assert manager.record(submitted.job_id).result is None
        finally:
            release.set()
            manager.wait(submitted.job_id, 3.0)

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
