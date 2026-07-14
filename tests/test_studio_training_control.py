# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio training parent-control tests

"""Exercise registry, platform reconciliation, cancellation, and SSE control."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest

from sc_neurocore.studio.platform.jobs import (
    StudioJobCancelled,
    StudioJobContext,
    StudioJobManager,
    StudioJobRecord,
)
from sc_neurocore.studio.training import (
    TrainingJob,
    _register_job,
    export_training_checkpoint,
    get_training_status,
    start_training,
    stop_training,
    stream_metrics,
)


def _manager(root: Path, *, timeout: float = 10.0) -> StudioJobManager:
    return StudioJobManager(
        root=root,
        allowed_kinds=frozenset({"training"}),
        default_timeout_seconds=timeout,
        max_artifact_bytes=10_000_000,
    )


def _wait_for_status(
    manager: StudioJobManager,
    job_id: str,
    statuses: set[str],
) -> StudioJobRecord:
    deadline = time.monotonic() + 10.0
    while True:
        record = manager.record(job_id)
        if record.status in statuses:
            return record
        if time.monotonic() >= deadline:
            pytest.fail(f"job {job_id} did not reach {sorted(statuses)}")
        time.sleep(0.01)


def _decode_sse(payload: str) -> dict[str, object]:
    return cast(dict[str, object], json.loads(payload.removeprefix("data: ").strip()))


def _submit_and_wait(
    manager: StudioJobManager,
    task: Callable[[StudioJobContext], dict[str, object]],
) -> StudioJobRecord:
    record = manager.submit(
        kind="training",
        owner="training-control-test",
        request_id=None,
        task=task,
    )
    return manager.wait(record.job_id, timeout_seconds=10.0)


def _register_proxy(
    record: StudioJobRecord, config: dict[str, object] | None = None
) -> TrainingJob:
    proxy = TrainingJob(dict(config or {}), job_id=record.job_id)
    proxy.status = "running"
    _register_job(proxy)
    return proxy


def test_local_job_survives_missing_platform_record_and_cancel_lookup(tmp_path: Path) -> None:
    """A legacy job remains observable when an unrelated manager lacks its ID."""
    result = start_training(
        {
            "dataset": "synthetic",
            "epochs": 1,
            "batch_size": 1024,
            "hidden": [4],
            "timesteps": 1,
        }
    )
    job_id = cast(str, result["job_id"])
    unrelated_manager = _manager(tmp_path / "unrelated")

    status = get_training_status(job_id, unrelated_manager)
    stopping = stop_training(job_id, unrelated_manager)

    assert status["job_id"] == job_id
    assert stopping == {"job_id": job_id, "status": "stopping"}
    deadline = time.monotonic() + 30.0
    while get_training_status(job_id)["status"] not in {"completed", "stopped", "failed"}:
        if time.monotonic() >= deadline:
            pytest.fail("legacy training did not terminate")
        time.sleep(0.01)
    first_stream = [_decode_sse(event) for event in stream_metrics(job_id)]
    assert first_stream[-1]["event"] in {"completed", "stopped", "error"}
    assert list(stream_metrics(job_id)) == []


def test_process_proxy_emits_heartbeat_when_manager_record_is_unavailable(
    tmp_path: Path,
) -> None:
    """A running proxy emits a heartbeat during transient manager lookup loss."""
    manager = _manager(tmp_path / "active", timeout=30.0)
    unrelated_manager = _manager(tmp_path / "empty")
    started = start_training(
        {
            "dataset": "synthetic",
            "epochs": 1_000_000,
            "batch_size": 1024,
            "hidden": [4],
            "timesteps": 1,
        },
        manager,
    )
    job_id = cast(str, started["job_id"])
    _wait_for_status(manager, job_id, {"running"})

    heartbeat = _decode_sse(next(stream_metrics(job_id, unrelated_manager)))
    stop_training(job_id, manager)
    terminal = manager.wait(job_id, timeout_seconds=30.0)

    assert heartbeat == {"event": "heartbeat"}
    assert terminal.status in {"cancelled", "completed"}


def test_unregistered_platform_records_map_to_public_status_and_sse(tmp_path: Path) -> None:
    """Persisted records remain observable after an in-memory registry restart."""
    manager = _manager(tmp_path / "records")

    def completed(_context: StudioJobContext) -> dict[str, object]:
        return {
            "final_metrics": {"train_accuracy": 0.75},
            "weight_checkpoint": {"format": "torch_state_dict"},
        }

    def completed_without_metrics(_context: StudioJobContext) -> dict[str, object]:
        return {"final_metrics": "invalid", "weight_checkpoint": "invalid"}

    def failed_empty(_context: StudioJobContext) -> dict[str, object]:
        raise RuntimeError

    def failed_message(_context: StudioJobContext) -> dict[str, object]:
        raise RuntimeError("operator-visible failure")

    def cancelled(_context: StudioJobContext) -> dict[str, object]:
        raise StudioJobCancelled("cancelled")

    records = {
        "completed": _submit_and_wait(manager, completed),
        "completed_empty": _submit_and_wait(manager, completed_without_metrics),
        "failed_empty": _submit_and_wait(manager, failed_empty),
        "failed_message": _submit_and_wait(manager, failed_message),
        "cancelled": _submit_and_wait(manager, cancelled),
    }

    completed_status = get_training_status(records["completed"].job_id, manager)
    invalid_status = get_training_status(records["completed_empty"].job_id, manager)
    failed_status = get_training_status(records["failed_message"].job_id, manager)
    cancelled_status = get_training_status(records["cancelled"].job_id, manager)
    persisted_stream = [
        _decode_sse(event) for event in stream_metrics(records["completed"].job_id, manager)
    ]
    missing_stream = [_decode_sse(event) for event in stream_metrics("sj_0000000000000000")]
    assert completed_status["status"] == "completed"
    assert completed_status["final_metrics"] == {"train_accuracy": 0.75}
    assert invalid_status["final_metrics"] is None
    assert invalid_status["weight_checkpoint"] is None
    assert failed_status["status"] == "failed"
    assert cancelled_status["status"] == "stopped"
    assert len(persisted_stream) == 1
    assert persisted_stream[0]["event"] == "completed"
    assert persisted_stream[0]["data"] == {"train_accuracy": 0.75}
    assert isinstance(persisted_stream[0]["timestamp"], float)
    assert missing_stream == [{"event": "error", "data": {"message": "Job not found"}}]


def test_proxy_reconciliation_updates_each_terminal_state_and_checkpoint(
    tmp_path: Path,
) -> None:
    """Registered proxies reconcile valid, malformed, failed, and cancelled records."""
    manager = _manager(tmp_path / "proxy-records")

    def completed(_context: StudioJobContext) -> dict[str, object]:
        return {
            "final_metrics": {"val_accuracy": 0.8},
            "weight_checkpoint": {"format": "torch_state_dict", "parameter_count": 10},
        }

    def malformed(_context: StudioJobContext) -> dict[str, object]:
        return {"final_metrics": 1, "weight_checkpoint": []}

    def failed(_context: StudioJobContext) -> dict[str, object]:
        raise RuntimeError("worker failed")

    def cancelled(_context: StudioJobContext) -> dict[str, object]:
        raise StudioJobCancelled("cancelled")

    records = [
        _submit_and_wait(manager, completed),
        _submit_and_wait(manager, malformed),
        _submit_and_wait(manager, failed),
        _submit_and_wait(manager, cancelled),
    ]
    proxies = [_register_proxy(record) for record in records]

    statuses = [get_training_status(record.job_id, manager) for record in records]
    malformed_checkpoint = export_training_checkpoint(records[1].job_id, manager)
    synthesized_events = [
        _decode_sse(event) for event in stream_metrics(records[0].job_id, manager)
    ]

    assert proxies[0].status == "completed"
    assert proxies[0].final_metrics == {"val_accuracy": 0.8}
    assert proxies[0].weight_checkpoint == {
        "format": "torch_state_dict",
        "parameter_count": 10,
    }
    assert proxies[1].status == "completed"
    assert proxies[1].final_metrics is None
    assert proxies[1].weight_checkpoint is None
    assert proxies[2].status == "failed"
    assert proxies[2].error == "worker failed"
    assert proxies[3].status == "stopped"
    assert statuses[2]["status"] == "failed"
    assert statuses[3]["status"] == "stopped"
    assert malformed_checkpoint["final_metrics"] is None
    assert len(synthesized_events) == 1
    assert synthesized_events[0]["event"] == "completed"
    assert synthesized_events[0]["data"] == {"val_accuracy": 0.8}
    assert isinstance(synthesized_events[0]["timestamp"], float)
