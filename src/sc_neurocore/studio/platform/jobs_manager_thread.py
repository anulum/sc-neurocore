# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio thread job supervision

"""Submission and bounded supervision for in-process Studio thread jobs."""

from __future__ import annotations

import secrets
import threading
from pathlib import Path

from sc_neurocore.studio.platform.jobs_context import StudioJobContext
from sc_neurocore.studio.platform.jobs_manager_state import _StudioJobManagerState
from sc_neurocore.studio.platform.jobs_models import (
    StudioJobCancelled,
    StudioJobRecord,
    StudioJobRejected,
    StudioJobTask,
)
from sc_neurocore.studio.platform.jobs_paths import _resolve_job_directory


def _submit_thread_job(
    manager: _StudioJobManagerState,
    *,
    kind: str,
    owner: str,
    request_id: str | None,
    task: StudioJobTask,
    timeout_seconds: float | None,
) -> StudioJobRecord:
    """Submit one local task to the bounded thread supervisor."""

    if kind not in manager._allowed_kinds:
        raise StudioJobRejected(f"Studio job kind '{kind}' is not allowed.")
    timeout = manager._default_timeout_seconds if timeout_seconds is None else timeout_seconds
    if timeout <= 0:
        raise StudioJobRejected("Studio job timeout must be positive.")
    job_id = f"sj_{secrets.token_hex(8)}"
    work_dir = _resolve_job_directory(
        root=manager._root,
        job_id=job_id,
        error_message="Studio job path escapes the job root.",
    )
    work_dir.mkdir(parents=True, exist_ok=False)
    cancel_event = threading.Event()
    done_event = threading.Event()
    record = StudioJobRecord(
        job_id=job_id,
        kind=kind,
        owner=owner,
        request_id=request_id,
        status="pending",
        execution_model="thread",
        created_at_utc=manager._timestamp_utc(),
    )
    with manager._lock:
        manager._records[job_id] = record
        manager._done_events[job_id] = done_event
        manager._cancel_events[job_id] = cancel_event
    supervisor = threading.Thread(
        target=manager._run_supervised,
        args=(job_id, work_dir, cancel_event, done_event, task, timeout),
        daemon=True,
    )
    supervisor.start()
    return record


def _run_thread_supervised(
    manager: _StudioJobManagerState,
    job_id: str,
    work_dir: Path,
    cancel_event: threading.Event,
    done_event: threading.Event,
    task: StudioJobTask,
    timeout_seconds: float,
) -> None:
    """Run one task in a daemon thread and persist its terminal state."""

    context = StudioJobContext(
        job_id=job_id,
        work_dir=work_dir,
        cancel_event=cancel_event,
        max_artifact_bytes=manager._max_artifact_bytes,
    )
    result_box: dict[str, dict[str, object]] = {}
    error_box: dict[str, BaseException] = {}
    manager._update(job_id, status="running", started_at_utc=manager._timestamp_utc())

    def target() -> None:
        try:
            result_box["result"] = task(context)
        except BaseException as exc:  # noqa: BLE001 - persisted as job failure state.
            error_box["error"] = exc

    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    worker.join(timeout_seconds)
    if worker.is_alive():
        cancel_event.set()
        worker.join(min(timeout_seconds, 1.0))
        manager._update(
            job_id,
            status="timed_out",
            error="Studio job exceeded its timeout.",
            finished_at_utc=manager._timestamp_utc(),
            artifacts=context.artifacts,
        )
        done_event.set()
        return
    error = error_box.get("error")
    if isinstance(error, StudioJobCancelled):
        manager._update(
            job_id,
            status="cancelled",
            finished_at_utc=manager._timestamp_utc(),
            artifacts=context.artifacts,
        )
    elif error is not None:
        manager._update(
            job_id,
            status="failed",
            error=str(error),
            finished_at_utc=manager._timestamp_utc(),
            artifacts=context.artifacts,
        )
    else:
        manager._update(
            job_id,
            status="completed",
            result=result_box.get("result", {}),
            finished_at_utc=manager._timestamp_utc(),
            artifacts=context.artifacts,
        )
    done_event.set()
