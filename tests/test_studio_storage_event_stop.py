# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Event worker cancellation and timeout

"""Stop actual event training without losing its full declared input or custody."""

import socket
import time
from collections.abc import Sequence
from typing import Literal

import pytest

from sc_neurocore.studio.platform import storage_generation_exchanges as exchanges
from sc_neurocore.studio.platform import storage_generation_supervisor as supervisor_module
from sc_neurocore.studio.platform.jobs_ledger import StudioJobLedger
from sc_neurocore.studio.platform.jobs_models import StudioJobRecord
from sc_neurocore.studio.platform.jobs_worker_recovery import worker_group_stopped
from sc_neurocore.studio.platform.storage_finish_client import exchange_finish as real_exchange
from sc_neurocore.studio.platform.storage_finish_protocol import (
    StorageFinishRequest,
    StorageFinishResponse,
)
from sc_neurocore.studio.platform.storage_isolated_jobs import IsolatedJobManager
from sc_neurocore.studio.platform.training_process import TRAINING_PROCESS_TASK
from tests.studio_storage_generation_runs import authority as authority
from tests.studio_storage_generation_runs import ledger as ledger
from tests.studio_storage_generation_support import Authority
from tests.test_studio_storage_event_training import base as base
from tests.test_studio_storage_event_training import launcher as launcher
from tests.test_studio_storage_event_training import event_config as event_config
from tests.test_studio_storage_isolated_jobs import manager as manager
from tests.test_studio_storage_isolated_jobs import request_on

#: Execution limit of the timeout case, in seconds, the same as cancellation's. The limit starts
#: when the generation supervisor begins to watch the granted worker, so it also covers the
#: worker's start-up and imports. Those take under 4 s on hosted runners (the whole cancel case
#: took 3.2-4.1 s in every Python lane of CI run 44278946068) but up to about a minute on a
#: host at load 50 (ML350, 2026-11-18), where a 10-second and then a 30-second limit lapsed
#: before the first training event. So the case does not wait for the limit: once the worker
#: is training, it moves the supervisor's clock past it (:class:`SupervisorClock`).
TIMEOUT_CASE_SECONDS = 120.0


class SupervisorClock:
    """The generation supervisor's monotonic clock, which a test may move forward.

    The supervisor reads ``time.monotonic`` to start the job's execution deadline and to check
    it. A test that puts this clock in place of the supervisor module's ``time`` can let the
    deadline lapse at a moment it chooses, after the worker's first training event, however
    long the worker took to start. Every other module keeps the real clock.
    """

    def __init__(self) -> None:
        self.advanced = 0.0

    def monotonic(self) -> float:
        """Return the real monotonic time plus the advance so far, in seconds."""
        return time.monotonic() + self.advanced

    def advance(self, seconds: float) -> None:
        """Move the clock forward.

        Parameters
        ----------
        seconds : float
            How far to move it, in seconds.
        """
        self.advanced += seconds


def exercise_event_stop(
    manager: IsolatedJobManager,
    ledger: StudioJobLedger,
    authority: Authority,
    event_config: dict[str, object],
    mode: Literal["cancel", "timeout"],
    *,
    timeout_seconds: float = TIMEOUT_CASE_SECONDS,
    clock: SupervisorClock | None = None,
) -> StudioJobRecord:
    """Stop a running worker and resolve a lost finish reply through real sockets.

    The authority applies the first finish before its reply is lost, so ``wait`` can return on
    the terminal record before the supervising generation repeats the finish; the finishes are
    counted only after the repeat has arrived.

    Parameters
    ----------
    manager, ledger, authority :
        Actual independently bounded API, SQLite and storage handler collaborators.
    event_config :
        Complete resolved event input, retained through the stop and replay.
    mode :
        Explicit cancellation or expiration of the worker's execution deadline.
    timeout_seconds :
        Execution budget for the timeout case; cancellation retains 120 seconds.
    clock :
        The supervisor's clock, in place, for the timeout case: once the worker is training,
        it is moved past the budget so the deadline lapses then. Cancellation does not use it.

    Returns
    -------
    StudioJobRecord
        Terminal record after confirmed group stop and identical replay.
    """
    config = {**event_config, "epochs": 10000}
    authority.lose["finish"] = 1
    with request_on("/api/training/start"):
        submitted = manager.submit_process_task(
            kind="training",
            owner="studio-training",
            request_id="event-stop",
            idempotency_key="event-stop-replay",
            task_path=TRAINING_PROCESS_TASK,
            payload=config,
            training_config=config,
            timeout_seconds=timeout_seconds if mode == "timeout" else 120.0,
        )
    deadline = time.monotonic() + 60
    with request_on("/api/studio/jobs", "GET"):
        while True:
            data, offset = manager.read_live_artifact_bytes(
                submitted.job_id, "training/events.jsonl", offset=0
            )
            if data:
                assert offset == len(data)
                break
            assert time.monotonic() < deadline
            status = manager.record(submitted.job_id).status
            assert status in {"pending", "running"}, (
                f"the job was {status} before its first training event "
                f"(mode {mode}; the timeout case's limit is {timeout_seconds:g} s)"
            )
            time.sleep(0.05)
        assert manager.record(submitted.job_id).status == "running"
    worker = (
        ledger.connection()
        .execute(
            "SELECT worker_identity,boot_id,group_id FROM job_workers WHERE job_id=?",
            (submitted.job_id,),
        )
        .fetchone()
    )
    assert worker is not None
    assert not worker_group_stopped(str(worker[0]), str(worker[1]), int(worker[2]))
    if mode == "timeout":
        assert clock is not None, "the timeout case needs the supervisor's clock in place"
        clock.advance(timeout_seconds)
    with request_on("/api/training/stop"):
        if mode == "cancel":
            assert manager.cancel(submitted.job_id).status == "cancelling"
        finished = manager.wait(submitted.job_id, timeout_seconds=60)
    assert finished.status == ("cancelled" if mode == "cancel" else "timed_out"), finished.error
    assert finished.training_config == config
    assert worker_group_stopped(str(worker[0]), str(worker[1]), int(worker[2]))
    # The authority applied the first finish and dropped its reply, so the record was terminal
    # before the supervising generation sent the identical repeat: wait for that repeat.
    deadline = time.monotonic() + 60
    while authority.seen.count("finish") < 2:
        assert time.monotonic() < deadline, "the finish whose reply was lost was not repeated"
        time.sleep(0.05)
    assert authority.seen.count("finish") == 2
    assert manager.generation_failures == {}
    with request_on("/api/studio/jobs/status", "GET"):
        assert manager.unreaped_workers == ()
        assert manager.status().active_count == 0
    with request_on("/api/studio/jobs", "GET"):
        assert manager.record(submitted.job_id) == finished
        assert manager.list_records() == (finished,)
    with request_on("/api/training/start"):
        replayed = manager.submit_process_task(
            kind="training",
            owner="studio-training",
            request_id="event-stop",
            idempotency_key="event-stop-replay",
            task_path=TRAINING_PROCESS_TASK,
            payload=config,
            training_config=config,
            timeout_seconds=timeout_seconds if mode == "timeout" else 120.0,
        )
    assert replayed == finished
    assert ledger.list_records() == (finished,)
    assert authority.seen.count("finish") == 2
    return finished


@pytest.mark.parametrize("mode", ["cancel", "timeout"])
def test_actual_event_stop_retains_complete_snapshot_and_proves_group_reaped(
    manager: IsolatedJobManager,
    ledger: StudioJobLedger,
    authority: Authority,
    event_config: dict[str, object],
    mode: Literal["cancel", "timeout"],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Terminal stop, identical retry and lost reply retain one durable event job.

    The timeout case lets the execution limit lapse once the worker is training, by moving the
    supervisor's clock, so a slow start cannot expire the job before its first event.

    Parameters
    ----------
    manager, ledger, authority, event_config :
        The collaborators of :func:`exercise_event_stop`.
    mode :
        Explicit cancellation or expiration of the worker's execution deadline.
    monkeypatch : pytest.MonkeyPatch
        Puts the supervisor's clock in place for the case.
    """
    clock = SupervisorClock()
    monkeypatch.setattr(supervisor_module, "time", clock)
    exercise_event_stop(manager, ledger, authority, event_config, mode, clock=clock)


def test_a_late_repeated_finish_is_awaited_before_the_finishes_are_counted(
    manager: IsolatedJobManager,
    ledger: StudioJobLedger,
    authority: Authority,
    event_config: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The record is terminal once the first finish is applied, before its repeat arrives.

    The authority applies the first finish and drops its reply, so ``wait`` can return on the
    terminal record while the supervising generation is still sending the identical repeat.
    Holding that repeat back for half a second makes the order certain: the stop is checked
    only after the authority has seen both finishes.

    Parameters
    ----------
    manager, ledger, authority, event_config :
        The collaborators of :func:`exercise_event_stop`.
    monkeypatch : pytest.MonkeyPatch
        Replaces the finish exchange with one that delays the second attempt.
    """
    attempts: list[float] = []

    def late_repeat(
        channel: socket.socket,
        request: StorageFinishRequest,
        payloads: Sequence[bytes],
        *,
        expected_service_uid: int,
        max_bytes: int,
        deadline: float,
    ) -> StorageFinishResponse:
        attempts.append(time.monotonic())
        if len(attempts) == 2:
            time.sleep(0.5)
            deadline += 0.5
        return real_exchange(
            channel,
            request,
            payloads,
            expected_service_uid=expected_service_uid,
            max_bytes=max_bytes,
            deadline=deadline,
        )

    monkeypatch.setattr(exchanges, "exchange_finish", late_repeat)
    exercise_event_stop(manager, ledger, authority, event_config, "cancel")
    assert len(attempts) == 2
