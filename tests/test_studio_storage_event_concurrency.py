# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Shared event training capacity and snapshot custody

"""Exercise two API managers against the same real event training authority."""

from pathlib import Path
import sqlite3
import time

import pytest

from sc_neurocore.studio.platform.jobs_admission import StudioJobQueueFull
from sc_neurocore.studio.platform.jobs_ledger import StudioJobLedger
from sc_neurocore.studio.platform.jobs_models import StudioJobRecord
from sc_neurocore.studio.platform.jobs_worker_recovery import worker_group_stopped
from sc_neurocore.studio.platform.storage_isolated_jobs import IsolatedJobManager
from sc_neurocore.studio.platform.training_process import TRAINING_PROCESS_TASK
from tests.studio_storage_generation_runs import api_runtime
from tests.studio_storage_generation_runs import authority as authority
from tests.studio_storage_generation_runs import ledger as ledger
from tests.studio_storage_generation_support import Authority
from tests.test_studio_storage_event_training import base as base
from tests.test_studio_storage_event_training import launcher as launcher
from tests.test_studio_storage_event_training import event_config as event_config
from tests.test_studio_storage_isolated_jobs import _configuration
from tests.test_studio_storage_isolated_jobs import manager as manager
from tests.test_studio_storage_isolated_jobs import request_on


def submit_event_training(
    manager: IsolatedJobManager, config: dict[str, object], key: str
) -> StudioJobRecord:
    """Admit the complete event contract through the delegated training route.

    Parameters
    ----------
    manager : IsolatedJobManager
        API facade bound to the real storage authority and launcher.
    config : dict
        Full resolved configuration, used as both worker input and custody.
    key : str
        Stable replay key of this intended run.

    Returns
    -------
    StudioJobRecord
        The admitted or identically replayed record.
    """
    with request_on("/api/training/start"):
        return manager.submit_process_task(
            kind="training",
            owner="studio-training",
            request_id=key,
            idempotency_key=key,
            task_path=TRAINING_PROCESS_TASK,
            payload=config,
            training_config=config,
            timeout_seconds=120.0,
        )


def test_two_api_managers_share_event_capacity_without_losing_snapshot_input(
    manager: IsolatedJobManager,
    ledger: StudioJobLedger,
    authority: Authority,
    event_config: dict[str, object],
    base: Path,
    tmp_path: Path,
) -> None:
    """Replay survives full capacity; reaped completion frees exactly one slot."""
    runtime = api_runtime(base, authority)
    peer = IsolatedJobManager(
        runtime,
        _configuration(base),
        allowed_kinds=frozenset({"training"}),
        default_timeout_seconds=120.0,
    )
    configs = [{**event_config, "epochs": 10000, "seed": seed} for seed in (41, 42)]
    try:
        first = submit_event_training(manager, configs[0], "event-concurrent-first")
        second = submit_event_training(peer, configs[1], "event-concurrent-second")
        assert first.job_id != second.job_id
        deadline = time.monotonic() + 60
        for owner, job in ((manager, first), (peer, second)):
            with request_on("/api/studio/jobs", "GET"):
                while True:
                    data, offset = owner.read_live_artifact_bytes(
                        job.job_id, "training/events.jsonl", offset=0
                    )
                    if data:
                        assert offset == len(data)
                        break
                    assert time.monotonic() < deadline
                    assert owner.record(job.job_id).status in {"pending", "running"}
                    time.sleep(0.05)
                assert owner.record(job.job_id).status == "running"
        workers = {
            str(row["job_id"]): (
                str(row["worker_identity"]),
                str(row["boot_id"]),
                int(row["group_id"]),
            )
            for row in ledger.connection().execute(
                "SELECT job_id,worker_identity,boot_id,group_id FROM job_workers"
            )
        }
        assert set(workers) == {first.job_id, second.job_id}
        assert not any(worker_group_stopped(*identity) for identity in workers.values())
        with pytest.raises(StudioJobQueueFull):
            submit_event_training(manager, event_config, "event-concurrent-third")
        assert (
            submit_event_training(peer, configs[0], "event-concurrent-first").job_id == first.job_id
        )
        assert len(ledger.list_records()) == 2

        snapshot_root = tmp_path / "active-snapshot"
        snapshot_root.mkdir(mode=0o700)
        with sqlite3.connect(snapshot_root / ledger.path.name) as destination:
            ledger.connection().backup(destination)
            assert destination.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        snapshot = StudioJobLedger(root=snapshot_root)
        try:
            assert snapshot.record(first.job_id).training_config == configs[0]
            assert snapshot.record(second.job_id).training_config == configs[1]
            assert {record.status for record in snapshot.list_records()} == {"running"}
        finally:
            snapshot.close()

        authority.lose["finish"] = 1
        with request_on("/api/training/stop"):
            manager.cancel(first.job_id)
            stopped = manager.wait(first.job_id, timeout_seconds=60)
            assert stopped.status == "cancelled"
            assert stopped.training_config == configs[0]
        assert worker_group_stopped(*workers[first.job_id])
        with request_on("/api/studio/jobs/status", "GET"):
            assert manager.status().active_count == 1
        with pytest.raises(StudioJobQueueFull) as retained_refusal:
            submit_event_training(manager, event_config, "event-concurrent-third")
        assert retained_refusal.value.running == 2
        third = submit_event_training(manager, event_config, "event-concurrent-third-new-attempt")
        with request_on("/api/training/start"):
            completed = manager.wait(third.job_id, timeout_seconds=120)
        assert completed.status == "completed", completed.error
        assert completed.training_config == event_config
        with request_on("/api/training/stop"):
            peer.cancel(second.job_id)
            stopped = peer.wait(second.job_id, timeout_seconds=60)
            assert stopped.status == "cancelled"
            assert stopped.training_config == configs[1]
        assert worker_group_stopped(*workers[second.job_id])
        assert len(ledger.list_records()) == 3
        with request_on("/api/studio/jobs/status", "GET"):
            assert manager.status().active_count == 0
            assert manager.unreaped_workers == peer.unreaped_workers == ()
        assert manager.generation_failures == peer.generation_failures == {}
    finally:
        runtime.live.close()
