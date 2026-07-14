# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio training parent-process control

"""Own the training registry, status, checkpoints, and metric stream."""

from __future__ import annotations

import json
import queue
import threading
from collections.abc import Iterator
from typing import Any, cast

from sc_neurocore.studio._training_events import (
    _event_from_platform_record,
    _read_live_training_events,
    _training_status_from_platform_status,
)
from sc_neurocore.studio._training_job import TrainingJob
from sc_neurocore.studio.platform.evidence_bundle import JsonValue
from sc_neurocore.studio.platform.jobs import (
    StudioJobManager,
    StudioJobRecord,
    StudioJobStatus,
)
from sc_neurocore.studio.platform.training_checkpoint import (
    build_training_checkpoint,
    import_training_checkpoint_payload,
)
from sc_neurocore.studio.platform.training_evidence import build_training_evidence_summary

_jobs: dict[str, TrainingJob] = {}
_jobs_lock = threading.Lock()


def _register_job(job: TrainingJob) -> None:
    """Register one parent-process job or process-worker proxy."""
    with _jobs_lock:
        _jobs[job.id] = job


def _get_registered_job(job_id: str) -> TrainingJob | None:
    """Return one registered job without exposing the mutable registry."""
    with _jobs_lock:
        return _jobs.get(job_id)


def _get_registered_pair(
    first_job_id: str,
    second_job_id: str,
) -> tuple[TrainingJob | None, TrainingJob | None]:
    """Return two registered jobs from one registry snapshot."""
    with _jobs_lock:
        return _jobs.get(first_job_id), _jobs.get(second_job_id)


def _start_training(
    config: dict[str, Any],
    job_manager: StudioJobManager | None = None,
) -> dict[str, Any]:
    """Start a process-backed or legacy-thread training job."""
    if job_manager is not None:
        from sc_neurocore.studio.platform.training_process import TRAINING_PROCESS_TASK

        record = job_manager.submit_process_task(
            kind="training",
            owner="studio-training",
            request_id=None,
            task_path=TRAINING_PROCESS_TASK,
            payload=config,
        )
        proxy = TrainingJob(config, job_id=record.job_id)
        proxy.status = "running"
        _register_job(proxy)
        return {"job_id": record.job_id, "status": "running"}

    job = TrainingJob(config)
    _register_job(job)
    job.start()
    return {"job_id": job.id, "status": "running"}


def _stop_training(
    job_id: str,
    job_manager: StudioJobManager | None = None,
) -> dict[str, Any]:
    """Request cooperative stop for a registered training job."""
    job = _get_registered_job(job_id)
    if job is None:
        return {"error": f"Job {job_id} not found"}
    job.stop()
    if job_manager is not None:
        try:
            job_manager.cancel(job_id)
        except KeyError:
            pass
    return {"job_id": job_id, "status": "stopping"}


def _get_training_status(
    job_id: str,
    job_manager: StudioJobManager | None = None,
) -> dict[str, Any]:
    """Return path-free status for a local job or persisted platform record."""
    job = _get_registered_job(job_id)
    if job is None:
        if job_manager is not None:
            try:
                record = job_manager.record(job_id)
            except KeyError:
                pass
            else:
                return _status_from_platform_record(
                    record,
                    evidence_summary=build_training_evidence_summary(
                        record,
                        job_manager.read_artifact,
                    ),
                )
        return {"error": f"Job {job_id} not found"}
    if job_manager is not None:
        try:
            record = job_manager.record(job_id)
        except KeyError:
            return job._public_status()
        _sync_proxy_job(job, record.status, record.error, record.result)
        return _status_with_evidence_summary(
            job._public_status(),
            build_training_evidence_summary(record, job_manager.read_artifact),
        )
    return job._public_status()


def _stream_metrics(
    job_id: str,
    job_manager: StudioJobManager | None = None,
) -> Iterator[str]:
    """Yield SSE-formatted training metric events for one job."""
    job = _get_registered_job(job_id)
    if job is None:
        if job_manager is not None:
            try:
                record = job_manager.record(job_id)
            except KeyError:
                record = None
            if record is not None:
                yield (
                    f"data: {json.dumps(_event_from_platform_record(record.status, record.error, record.result))}\n\n"
                )
                return
        yield f"data: {json.dumps({'event': 'error', 'data': {'message': 'Job not found'}})}\n\n"
        return

    live_event_offset = 0
    live_event_buffer = ""
    live_terminal_seen = False
    while True:
        if job_manager is not None:
            try:
                record = job_manager.record(job_id)
            except KeyError:
                record = None
            if record is not None:
                _sync_proxy_job(job, record.status, record.error, record.result)
                live_events, live_event_offset, live_event_buffer = _read_live_training_events(
                    job_manager,
                    job_id,
                    offset=live_event_offset,
                    buffer=live_event_buffer,
                )
                for event in live_events:
                    if event.get("event") in ("completed", "stopped", "error"):
                        live_terminal_seen = True
                    yield f"data: {json.dumps(event)}\n\n"
                if live_terminal_seen:
                    break
                if job.status in ("completed", "stopped", "failed"):
                    yield (
                        "data: "
                        f"{json.dumps(_event_from_platform_record(record.status, record.error, record.result))}\n\n"
                    )
                    break
        try:
            event = job.metrics.get(timeout=1.0)
            yield f"data: {json.dumps(event)}\n\n"
            if event["event"] in ("completed", "stopped", "error"):
                break
        except queue.Empty:
            if job.status in ("completed", "stopped", "failed"):
                break
            yield f"data: {json.dumps({'event': 'heartbeat'})}\n\n"


def _list_jobs() -> list[dict[str, Any]]:
    """Return path-free summaries for all registered training jobs."""
    with _jobs_lock:
        return [
            {"job_id": job.id, "status": job.status, "config": job.config} for job in _jobs.values()
        ]


def _export_training_checkpoint(
    job_id: str,
    job_manager: StudioJobManager | None = None,
) -> dict[str, Any]:
    """Build a portable checkpoint for one registered training job."""
    job = _get_registered_job(job_id)
    if job is None:
        return {"error": f"Job {job_id} not found"}
    status = _get_training_status(job_id, job_manager)
    final_metrics = status.get("final_metrics")
    evidence_summary = status.get("evidence_summary")
    weight_checkpoint = status.get("weight_checkpoint")
    checkpoint = build_training_checkpoint(
        job_id=job_id,
        config=job.config,
        status=str(status.get("status", job.status)),
        final_metrics=final_metrics if isinstance(final_metrics, dict) else None,
        evidence_summary=evidence_summary if isinstance(evidence_summary, dict) else None,
        weight_checkpoint=weight_checkpoint if isinstance(weight_checkpoint, dict) else None,
    )
    return checkpoint.to_public_dict()


def _import_training_checkpoint(data: dict[str, Any]) -> dict[str, Any]:
    """Validate and import one portable Training Monitor checkpoint."""
    return import_training_checkpoint_payload(data)


def _sync_proxy_job(
    job: TrainingJob,
    platform_status: StudioJobStatus,
    platform_error: str | None,
    platform_result: dict[str, object] | None,
) -> None:
    """Update a parent-process proxy from platform terminal state."""
    if platform_status == "completed":
        job.status = "completed"
        final_metrics = (platform_result or {}).get("final_metrics")
        if isinstance(final_metrics, dict):
            job.final_metrics = dict(final_metrics)
        weight_checkpoint = (platform_result or {}).get("weight_checkpoint")
        if isinstance(weight_checkpoint, dict):
            job.weight_checkpoint = cast(dict[str, JsonValue], dict(weight_checkpoint))
        return
    if platform_status in ("cancelled", "cancelling", "timed_out"):
        job.status = "stopped"
        job.error = platform_error
        return
    if platform_status == "failed":
        job.status = "failed"
        job.error = platform_error


def _status_from_platform_record(
    record: StudioJobRecord,
    *,
    evidence_summary: dict[str, object] | None = None,
) -> dict[str, Any]:
    """Return Training Monitor status synthesized from a platform record."""
    platform_result = record.result if isinstance(record.result, dict) else None
    final_metrics = (platform_result or {}).get("final_metrics")
    weight_checkpoint = (platform_result or {}).get("weight_checkpoint")
    return _status_with_evidence_summary(
        {
            "error": record.error,
            "final_metrics": final_metrics if isinstance(final_metrics, dict) else None,
            "job_id": record.job_id,
            "status": _training_status_from_platform_status(record.status),
            "weight_checkpoint": weight_checkpoint if isinstance(weight_checkpoint, dict) else None,
        },
        evidence_summary,
    )


def _status_with_evidence_summary(
    status: dict[str, Any],
    evidence_summary: dict[str, object] | None,
) -> dict[str, Any]:
    """Attach path-free evidence metadata to a public training status."""
    if evidence_summary is None:
        return status
    return {**status, "evidence_summary": evidence_summary}
