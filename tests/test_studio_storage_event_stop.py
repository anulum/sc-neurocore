# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Event worker cancellation and timeout

"""Stop actual event training without losing its full declared input or custody."""

import time
from typing import Literal

import pytest

from sc_neurocore.studio.platform.jobs_ledger import StudioJobLedger
from sc_neurocore.studio.platform.jobs_models import StudioJobRecord
from sc_neurocore.studio.platform.jobs_worker_recovery import worker_group_stopped
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


def exercise_event_stop(
    manager: IsolatedJobManager,
    ledger: StudioJobLedger,
    authority: Authority,
    event_config: dict[str, object],
    mode: Literal["cancel", "timeout"],
    *,
    timeout_seconds: float = 10.0,
) -> StudioJobRecord:
    """Stop a running worker and resolve a lost finish reply through real sockets.

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
            assert manager.record(submitted.job_id).status in {"pending", "running"}
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
    with request_on("/api/training/stop"):
        if mode == "cancel":
            assert manager.cancel(submitted.job_id).status == "cancelling"
        finished = manager.wait(submitted.job_id, timeout_seconds=60)
    assert finished.status == ("cancelled" if mode == "cancel" else "timed_out"), finished.error
    assert finished.training_config == config
    assert worker_group_stopped(str(worker[0]), str(worker[1]), int(worker[2]))
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
) -> None:
    """Terminal stop, identical retry and lost reply retain one durable event job."""
    exercise_event_stop(manager, ledger, authority, event_config, mode)
