# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Catalogue scan process contracts

"""Real catalogue results and refusals across the named worker boundary."""

from pathlib import Path

import pytest

from sc_neurocore.studio.model_scan import scan_all_models
from sc_neurocore.studio.platform.jobs_manager import StudioJobManager

TASK = "sc_neurocore.studio.api.model_scan_jobs:execute_model_scan_process_task"


def test_process_preserves_complete_scan_results_and_digests(tmp_path: Path) -> None:
    """A short full-catalogue scan preserves every row, failure and manifest field."""
    expected = scan_all_models(current=10.0, duration=1.0)
    manager = StudioJobManager(
        root=tmp_path, allowed_kinds=frozenset({"model_scan"}), default_timeout_seconds=30.0
    )
    try:
        job = manager.submit_process_task(
            kind="model_scan",
            owner="studio",
            request_id=None,
            task_path=TASK,
            payload={"current": 10.0, "duration": 1.0},
        )
        result = manager.wait(job.job_id, 30.0)
        assert result.status == "completed", result.error
        assert result.execution_model == "process"
        assert result.owner == "studio"
        assert result.result == expected
    finally:
        for record in manager.list_records():
            if record.status not in {"completed", "failed", "timed_out", "cancelled"}:
                manager.cancel(record.job_id)
                manager.wait(record.job_id, 15.0)
        manager._ledger.close()


@pytest.mark.parametrize(
    "payload",
    [
        {"current": 10.0},
        {"current": 10.0, "duration": 0.0},
        {"current": 10.0, "duration": float("inf")},
        {"current": float("nan"), "duration": 1.0},
        {"current": True, "duration": 1.0},
        {"current": "10", "duration": 1.0},
        {"current": 10.0, "duration": 1.0, "models": ["AdExNeuron"]},
    ],
)
def test_worker_refuses_invalid_scan_envelopes(tmp_path: Path, payload: dict[str, object]) -> None:
    """Malformed inputs cannot produce successful or silently subsetted catalogue evidence."""
    manager = StudioJobManager(
        root=tmp_path, allowed_kinds=frozenset({"model_scan"}), default_timeout_seconds=30.0
    )
    try:
        job = manager.submit_process_task(
            kind="model_scan", owner="studio", request_id=None, task_path=TASK, payload=payload
        )
        result = manager.wait(job.job_id, 30.0)
        assert result.status == "failed", result.error
        assert result.result is None
        assert result.error == "ValidationError"
    finally:
        for record in manager.list_records():
            if record.status not in {"completed", "failed", "timed_out", "cancelled"}:
                manager.cancel(record.job_id)
                manager.wait(record.job_id, 15.0)
        manager._ledger.close()
