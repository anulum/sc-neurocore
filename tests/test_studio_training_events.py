# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio training event-stream tests

"""Exercise persisted event normalization and public SSE recovery paths."""

from __future__ import annotations

import json
import math
import threading
import time
from pathlib import Path
from typing import cast

import pytest

from sc_neurocore.studio.platform.jobs import (
    StudioJobCancelled,
    StudioJobContext,
    StudioJobManager,
    StudioJobRecord,
    StudioJobTask,
)
from sc_neurocore.studio.training import (
    TRAINING_EVENT_LOG_ARTIFACT_PATH,
    TrainingJob,
    _register_job,
    stream_metrics,
)


def _manager(root: Path, *, timeout: float = 10.0) -> StudioJobManager:
    return StudioJobManager(
        root=root,
        allowed_kinds=frozenset({"training"}),
        default_timeout_seconds=timeout,
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


def _context(tmp_path: Path, job_id: str) -> StudioJobContext:
    work_dir = tmp_path / job_id
    work_dir.mkdir()
    return StudioJobContext(
        job_id=job_id,
        work_dir=work_dir,
        cancel_event=threading.Event(),
        max_artifact_bytes=50_000_000,
    )


def test_platform_record_stream_maps_every_terminal_event_shape(tmp_path: Path) -> None:
    """Persisted platform states map to stable completed, error, and stopped SSE."""
    manager = _manager(tmp_path / "terminal-records")

    def completed(_context: StudioJobContext) -> dict[str, object]:
        return {"final_metrics": {"train_accuracy": 0.75}}

    def completed_without_metrics(_context: StudioJobContext) -> dict[str, object]:
        return {"final_metrics": "invalid"}

    def failed_empty(_context: StudioJobContext) -> dict[str, object]:
        raise RuntimeError

    def failed_message(_context: StudioJobContext) -> dict[str, object]:
        raise RuntimeError("operator-visible failure")

    def cancelled(_context: StudioJobContext) -> dict[str, object]:
        raise StudioJobCancelled("cancelled")

    def submit(task: StudioJobTask) -> StudioJobRecord:
        record = manager.submit(
            kind="training",
            owner="training-events-test",
            request_id=None,
            task=task,
        )
        return manager.wait(record.job_id, timeout_seconds=10.0)

    records = {
        "completed": submit(completed),
        "completed_empty": submit(completed_without_metrics),
        "failed_empty": submit(failed_empty),
        "failed_message": submit(failed_message),
        "cancelled": submit(cancelled),
    }

    timeout_manager = _manager(tmp_path / "timed-out", timeout=0.01)

    def slow(_context: StudioJobContext) -> dict[str, object]:
        time.sleep(0.1)
        return {}

    timed_out = timeout_manager.submit(
        kind="training",
        owner="training-events-test",
        request_id=None,
        task=slow,
    )
    timed_out = timeout_manager.wait(timed_out.job_id, timeout_seconds=1.0)

    release = threading.Event()

    def running(_context: StudioJobContext) -> dict[str, object]:
        release.wait(timeout=5.0)
        return {}

    running_record = manager.submit(
        kind="training",
        owner="training-events-test",
        request_id=None,
        task=running,
    )
    _wait_for_status(manager, running_record.job_id, {"running"})

    events = {
        name: _decode_sse(next(stream_metrics(record.job_id, manager)))
        for name, record in records.items()
    }
    events["timed_out"] = _decode_sse(next(stream_metrics(timed_out.job_id, timeout_manager)))
    events["running"] = _decode_sse(next(stream_metrics(running_record.job_id, manager)))
    release.set()
    manager.wait(running_record.job_id, timeout_seconds=10.0)

    assert events["completed"]["event"] == "completed"
    assert events["completed"]["data"] == {"train_accuracy": 0.75}
    assert events["completed_empty"]["data"] == {}
    assert events["failed_empty"]["data"] == {"message": "Training failed."}
    assert events["failed_message"]["data"] == {"message": "operator-visible failure"}
    assert events["cancelled"]["data"] == {"message": "Training stopped."}
    assert events["timed_out"]["data"] == {"message": "Studio job exceeded its timeout."}
    assert events["running"] == {"event": "heartbeat"}
    for name in ("completed", "failed_empty", "failed_message", "cancelled", "timed_out"):
        assert math.isfinite(cast(float, events[name]["timestamp"]))


def test_stream_tolerates_partial_and_malformed_live_event_rows(tmp_path: Path) -> None:
    """The public SSE stream skips corrupt rows and joins a split JSONL record."""
    manager_root = tmp_path / "live-events"
    manager = _manager(manager_root)
    release = threading.Event()

    def running(_context: StudioJobContext) -> dict[str, object]:
        release.wait(timeout=5.0)
        return {"final_metrics": {"train_accuracy": 1.0}}

    record = manager.submit(
        kind="training",
        owner="training-events-test",
        request_id=None,
        task=running,
    )
    _wait_for_status(manager, record.job_id, {"running"})
    proxy = TrainingJob({}, job_id=record.job_id)
    proxy.status = "running"
    _register_job(proxy)
    event_path = manager_root / record.job_id / TRAINING_EVENT_LOG_ARTIFACT_PATH
    event_path.parent.mkdir(parents=True, exist_ok=True)
    event_path.write_bytes(
        b'\nnot-json\n[]\n{"event":"epoch","data":{"epoch":0}}\n{"event":"completed"'
    )
    stream = stream_metrics(record.job_id, manager)

    epoch_event = _decode_sse(next(stream))
    heartbeat = _decode_sse(next(stream))
    with event_path.open("ab") as handle:
        handle.write(b',"data":{}}\n')
    completed_event = _decode_sse(next(stream))
    release.set()
    manager.wait(record.job_id, timeout_seconds=10.0)

    assert epoch_event == {"event": "epoch", "data": {"epoch": 0}}
    assert heartbeat == {"event": "heartbeat"}
    assert completed_event == {"event": "completed", "data": {}}
    assert list(stream) == []


@pytest.mark.parametrize(
    ("dataset", "expected"),
    [
        (("synthetic",), ["synthetic"]),
        (float("nan"), None),
        (Path("synthetic"), "synthetic"),
    ],
)
def test_persisted_events_normalize_library_values(
    tmp_path: Path,
    dataset: object,
    expected: object,
) -> None:
    """Persisted worker events remain JSON-safe for supported library values."""
    pytest.importorskip("torch")
    context = _context(tmp_path, f"sj_training_json_{type(dataset).__name__}")
    events: list[dict[str, object]] = []

    def event_sink(event: dict[str, object]) -> None:
        events.append(event)
        context.append_artifact_event(TRAINING_EVENT_LOG_ARTIFACT_PATH, event)

    job = TrainingJob(
        {
            "dataset": dataset,
            "epochs": 1,
            "batch_size": 64,
            "hidden": [8],
            "timesteps": 1,
        },
        job_id=context.job_id,
        cancelled=lambda: True,
        event_sink=event_sink,
    )

    with pytest.raises(StudioJobCancelled, match="stopped"):
        job.run_blocking(context)

    config_event = events[0]
    data = cast(dict[str, object], config_event["data"])
    assert data["dataset"] == expected
    assert math.isfinite(cast(float, config_event["timestamp"]))
