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

Observation failures are real: another connection corrupts the job's stored
artifacts, so the owning supervisor's next ledger read fails; a refused
terminal write is a real SQLite trigger. The ledger, the transitions and the
managers are the production ones.
"""

from __future__ import annotations

import threading
import os
import time
import sqlite3
from contextlib import closing

import pytest

from sc_neurocore.studio.platform.jobs import (
    StudioJobContext,
    StudioJobManager,
)


from tests.test_studio_jobs_cancel_race import manager as manager


def _corrupt_artifacts(manager: StudioJobManager, job_id: str) -> None:
    """Damage the stored artifacts of one job through a second real connection."""
    with closing(sqlite3.connect(manager.ledger_path, isolation_level=None)) as other:
        other.execute("UPDATE jobs SET artifacts='[{}]' WHERE job_id=?", (job_id,))


def _status(manager: StudioJobManager, job_id: str) -> str:
    """Read the stored status without decoding the rest of a possibly damaged row."""
    with closing(sqlite3.connect(manager.ledger_path)) as observer:
        row = observer.execute("SELECT status FROM jobs WHERE job_id=?", (job_id,)).fetchone()
    return str(row[0])


class TestPeerCancellation:
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
        self, manager: StudioJobManager
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
            _corrupt_artifacts(manager, submitted.job_id)
            # The damaged row is readable again once the supervisor settled it.
            assert manager._done_events[submitted.job_id].wait(5.0)
            outcome = manager.record(submitted.job_id)
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
            previous_hook = threading.excepthook
            if write_failure:

                def capture_error(args: threading.ExceptHookArgs) -> None:
                    # Observes the supervisor thread's unhandled error only.
                    if args.exc_value is not None:
                        errors.append(args.exc_value)
                    failed.set()

                threading.excepthook = capture_error
                with closing(sqlite3.connect(manager.ledger_path, isolation_level=None)) as other:
                    other.execute(
                        "CREATE TRIGGER refuse_failure BEFORE UPDATE OF status ON jobs "
                        "WHEN NEW.status = 'failed' "
                        "BEGIN SELECT RAISE(ABORT, 'terminal write refused'); END"
                    )
            try:
                if read_failure:
                    _corrupt_artifacts(manager, submitted.job_id)
                else:
                    assert peer.cancel(submitted.job_id).status in {"cancelling", "cancelled"}
                if write_failure:
                    assert failed.wait(5.0)
                    assert len(errors) == 1
                    assert isinstance(errors[0], sqlite3.IntegrityError)
                    assert str(errors[0]) == "terminal write refused"
                    assert manager._done_events[submitted.job_id].is_set()
                    assert _status(manager, submitted.job_id) == "running"
                    with pytest.raises(ProcessLookupError):
                        os.kill(pid, 0)
                    return
            finally:
                threading.excepthook = previous_hook
                if write_failure:
                    with closing(
                        sqlite3.connect(manager.ledger_path, isolation_level=None)
                    ) as other:
                        other.execute("DROP TRIGGER refuse_failure")
                        other.execute(
                            "UPDATE jobs SET artifacts='[]' WHERE job_id=?", (submitted.job_id,)
                        )
            assert manager._done_events[submitted.job_id].wait(5.0)
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
        self, manager: StudioJobManager, read_failure: bool
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
                _corrupt_artifacts(manager, submitted.job_id)
            else:
                requested = peer.cancel(submitted.job_id)
                assert requested.status in {"cancelling", "cancelled"}
            assert manager._done_events[submitted.job_id].wait(5.0)
            outcome = peer.wait(submitted.job_id, 2.0)
            assert outcome.status == ("failed" if read_failure else "cancelled")
            if read_failure:
                assert outcome.error is not None and "Worker stopped: True" in outcome.error
            assert manager.record(submitted.job_id).result is None
        finally:
            release.set()
            manager.wait(submitted.job_id, 3.0)
