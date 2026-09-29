# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Admission replay after event archive purge

"""Preserve exact admission outcomes without resurrecting purged event runs."""

import json

import pytest

from sc_neurocore.studio.platform.jobs_ledger import StudioJobLedger
from sc_neurocore.studio.platform.storage_isolated_jobs import IsolatedJobManager
from sc_neurocore.studio.platform.training_process import TRAINING_PROCESS_TASK
from tests.studio_storage_generation_runs import authority as authority
from tests.studio_storage_generation_runs import ledger as ledger
from tests.studio_storage_generation_support import Authority
from tests.test_studio_storage_event_stop import exercise_event_stop
from tests.test_studio_storage_event_training import base as base
from tests.test_studio_storage_event_training import launcher as launcher
from tests.test_studio_storage_event_training import event_config as event_config
from tests.test_studio_storage_isolated_jobs import manager as manager
from tests.test_studio_storage_isolated_jobs import request_on


def test_original_event_admission_after_purge_cannot_recreate_its_job(
    manager: IsolatedJobManager,
    ledger: StudioJobLedger,
    authority: Authority,
    event_config: dict[str, object],
) -> None:
    """The old key returns not found; only an explicit new admission starts work."""
    stopped = exercise_event_stop(manager, ledger, authority, event_config, "cancel")
    original = (
        ledger.connection()
        .execute("SELECT payload_sha256,record_json FROM storage_admission_replays")
        .fetchone()
    )
    assert original is not None
    original_digest, original_snapshot = str(original[0]), str(original[1])
    pending = json.loads(original_snapshot)
    assert pending["job_id"] == stopped.job_id
    assert pending["training_config"] == stopped.training_config
    with request_on("/api/studio/audit/quarantine/archive/purge"):
        assert manager.purge_terminal_record(stopped.job_id) == stopped
    finishes = authority.seen.count("finish")
    assert stopped.training_config is not None
    with request_on("/api/training/start"), pytest.raises(KeyError):
        manager.submit_process_task(
            kind="training",
            owner="studio-training",
            request_id="event-stop",
            idempotency_key="event-stop-replay",
            task_path=TRAINING_PROCESS_TASK,
            payload=stopped.training_config,
            training_config=stopped.training_config,
            timeout_seconds=120.0,
        )
    assert ledger.list_records() == ()
    assert ledger.connection().execute("SELECT COUNT(*) FROM job_workers").fetchone()[0] == 0
    assert (
        ledger.connection().execute("SELECT COUNT(*) FROM admission_reservations").fetchone()[0]
        == 0
    )
    assert authority.seen.count("finish") == finishes
    assert not (ledger.path.parent / stopped.job_id).exists()
    retained = (
        ledger.connection()
        .execute("SELECT payload_sha256,record_json FROM storage_admission_replays")
        .fetchone()
    )
    assert tuple(retained) == (original_digest, original_snapshot)
    with request_on("/api/training/start"):
        new = manager.submit_process_task(
            kind="training",
            owner="studio-training",
            request_id="new-after-purge",
            idempotency_key="new-after-purge",
            task_path=TRAINING_PROCESS_TASK,
            payload=event_config,
            training_config=event_config,
            timeout_seconds=120.0,
        )
        completed = manager.wait(new.job_id, timeout_seconds=120)
    assert new.job_id != stopped.job_id
    assert completed.status == "completed", completed.error
    assert completed.training_config == event_config
    assert ledger.list_records() == (completed,)
    assert not (ledger.path.parent / stopped.job_id).exists()
    with request_on("/api/studio/jobs/status", "GET"):
        assert manager.status().active_count == 0
        assert manager.unreaped_workers == ()
    assert manager.generation_failures == {}
