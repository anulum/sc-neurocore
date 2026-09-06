# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio job records and artifacts

"""Cancellation, records, status, purge, and artifact reads for Studio jobs."""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path

from sc_neurocore.studio.platform.jobs_ledger_schema import TERMINAL_STATUSES
from sc_neurocore.studio.platform.jobs_manager_state import _StudioJobManagerState
from sc_neurocore.studio.platform.jobs_models import (
    StudioJobArtifact,
    StudioJobArtifactPayload,
    StudioJobArtifactUnavailable,
    StudioJobListSnapshot,
    StudioJobRecord,
    StudioJobRejected,
    StudioJobResourceProfile,
    StudioJobStatus,
    StudioJobStatusSnapshot,
)
from sc_neurocore.studio.platform.jobs_paths import (
    _find_artifact,
    _is_confined_path,
    _normalize_artifact_lookup_path,
    _relative_path_candidate,
    _resolve_confined_child,
)


def _cancel_job(manager: _StudioJobManagerState, job_id: str) -> StudioJobRecord:
    """Request cooperative cancellation for one job.

    A job this process is not supervising can still be asked to cancel: the
    request is recorded durably, and the supervisor that owns it acts on it.

    Cancelling a job that has already stopped is a no-op rather than an error,
    and that has to hold even when the job stops between the status read and
    the write. The pre-check reads outside the ledger transaction, so under
    load a run can reach ``cancelled`` in that gap; the ledger then refuses
    ``cancelled -> cancelling`` and the refusal reached the operator as a
    server error for pressing Stop on a run that had just finished. Losing
    that race means the job is already stopped, so the record it reached is
    returned. A refusal for any other reason still propagates.
    """

    record = manager._ledger.record(job_id)
    if record.status in TERMINAL_STATUSES or record.status == "cancelling":
        return record
    with manager._lock:
        cancel_event = manager._cancel_events.get(job_id)
    if cancel_event is not None:
        cancel_event.set()
    try:
        return manager._ledger.transition(job_id, "cancelling", reason="cancellation requested")
    except StudioJobRejected:
        settled = manager._ledger.record(job_id)
        if settled.status in TERMINAL_STATUSES or settled.status == "cancelling":
            return settled
        raise


def _wait_for_job(
    manager: _StudioJobManagerState,
    job_id: str,
    timeout_seconds: float | None,
) -> StudioJobRecord:
    """Wait for one job and return its latest immutable record."""

    with manager._lock:
        done_event = manager._done_events[job_id]
    done_event.wait(timeout_seconds)
    return manager.record(job_id)


def _get_job_record(
    manager: _StudioJobManagerState,
    job_id: str,
    *,
    actor: str | None = None,
    workspace: str | None = None,
) -> StudioJobRecord:
    """Return the latest durable record for one job.

    A job belonging to another actor or workspace raises ``KeyError``, so an
    isolated caller cannot distinguish it from one that never existed.
    """

    return manager._ledger.record(job_id, actor=actor, workspace=workspace)


def _list_job_records(
    manager: _StudioJobManagerState,
    *,
    actor: str | None = None,
    workspace: str | None = None,
) -> tuple[StudioJobRecord, ...]:
    """Return all durable jobs in creation order, scoped when asked."""

    return manager._ledger.list_records(actor=actor, workspace=workspace)


def _list_job_snapshot(
    manager: _StudioJobManagerState,
    *,
    actor: str | None = None,
    workspace: str | None = None,
) -> StudioJobListSnapshot:
    """Return a path-free snapshot of every job visible to the caller."""

    return StudioJobListSnapshot(records=manager.list_records(actor=actor, workspace=workspace))


def _purge_terminal_job(manager: _StudioJobManagerState, job_id: str) -> StudioJobRecord:
    """Delete one terminal job directory and its in-memory state."""

    record = manager._ledger.record(job_id)
    if record.status not in TERMINAL_STATUSES:
        raise StudioJobRejected("Studio active jobs cannot be purged.")
    try:
        work_dir = manager._job_work_dir(record.job_id)
    except ValueError as exc:
        raise StudioJobRejected(str(exc)) from exc
    if work_dir.exists():
        if not work_dir.is_dir():
            raise StudioJobRejected("Studio job purge target is not a directory.")
        shutil.rmtree(work_dir)
    manager._ledger.delete(job_id)
    with manager._lock:
        manager._done_events.pop(job_id, None)
        manager._cancel_events.pop(job_id, None)
    return record


def _commit_supervised_update(
    manager: _StudioJobManagerState,
    job_id: str,
    *,
    status: StudioJobStatus,
    started_at_utc: str | None = None,
    finished_at_utc: str | None = None,
    error: str | None = None,
    result: dict[str, object] | None = None,
    artifacts: tuple[StudioJobArtifact, ...] | None = None,
) -> None:
    """Commit one supervised state change to the durable ledger.

    The artifact manifest lands in the same transaction as the terminal status,
    so a crash between writing an artifact and committing the outcome leaves the
    job recoverable rather than falsely complete.

    A job asked to cancel before its supervisor marked it running keeps its
    ``cancelling`` status; that rule lives inside the ledger transaction, where
    it cannot race the cancellation it is reconciling with.
    """

    record = manager._ledger.transition(
        job_id,
        status,
        reason="supervised",
        started_at_utc=started_at_utc,
        finished_at_utc=finished_at_utc,
        error=error,
        result=result,
        artifacts=artifacts,
    )
    if record.status in TERMINAL_STATUSES:
        # The job stopped occupying a slot the moment its outcome was
        # committed, not when the supervisor thread happens to unwind.
        manager._admission.release()


def _read_declared_artifact(
    manager: _StudioJobManagerState,
    job_id: str,
    relative_path: str,
) -> StudioJobArtifactPayload:
    """Return one manifest-declared payload after size and hash validation."""

    record = manager.record(job_id)
    requested_path = _normalize_artifact_lookup_path(relative_path)
    artifact = _find_artifact(record.artifacts, requested_path)
    try:
        artifact_path = _resolve_confined_child(
            root=manager._job_work_dir(record.job_id),
            relative_path=artifact.relative_path,
            error_message="Studio job artifact path escapes the job directory.",
        )
    except ValueError as exc:
        raise StudioJobArtifactUnavailable(str(exc)) from exc
    if not artifact_path.is_file():
        raise StudioJobArtifactUnavailable("Studio job artifact is unavailable.")
    payload = artifact_path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if len(payload) != artifact.size_bytes or digest != artifact.sha256:
        raise StudioJobArtifactUnavailable("Studio job artifact integrity check failed.")
    return StudioJobArtifactPayload(artifact=artifact, payload=payload)


def _read_live_artifact(
    manager: _StudioJobManagerState,
    job_id: str,
    relative_path: str,
    *,
    offset: int,
    max_bytes: int,
) -> tuple[bytes, int]:
    """Return a bounded newly appended slice from one confined live artifact."""

    manager.record(job_id)
    if offset < 0:
        raise ValueError("Studio live artifact offset must be non-negative.")
    if max_bytes <= 0:
        raise ValueError("Studio live artifact read size must be positive.")
    requested_path = _normalize_artifact_lookup_path(relative_path)
    try:
        work_dir = manager._job_work_dir(job_id)
        candidate = _relative_path_candidate(
            requested_path,
            error_message="Studio job artifact path escapes the job directory.",
        )
        resolved_root = os.path.realpath(os.fspath(work_dir))
        resolved_artifact_path = os.path.realpath(os.path.join(resolved_root, os.fspath(candidate)))
        if not _is_confined_path(root=resolved_root, child=resolved_artifact_path):
            raise ValueError("Studio job artifact path escapes the job directory.")
        artifact_path = Path(resolved_artifact_path)
    except ValueError as exc:
        raise StudioJobArtifactUnavailable(str(exc)) from exc
    if not artifact_path.is_file():
        return b"", offset
    with artifact_path.open("rb") as handle:
        handle.seek(offset)
        payload = handle.read(max_bytes)
        return payload, offset + len(payload)


def _job_manager_status(manager: _StudioJobManagerState) -> StudioJobStatusSnapshot:
    """Return aggregate path-free manager health and resource profiles."""

    records = manager.list_records()
    active_statuses = {"pending", "running", "cancelling"}
    allowed_kinds = tuple(sorted(manager._allowed_kinds))
    return StudioJobStatusSnapshot(
        configured=manager._configured,
        allowed_kinds=allowed_kinds,
        active_count=sum(record.status in active_statuses for record in records),
        completed_count=sum(record.status == "completed" for record in records),
        failed_count=sum(record.status == "failed" for record in records),
        process_count=sum(record.execution_model == "process" for record in records),
        thread_count=sum(record.execution_model == "thread" for record in records),
        timed_out_count=sum(record.status == "timed_out" for record in records),
        interrupted_count=sum(record.status == "interrupted" for record in records),
        unknown_count=sum(record.status == "unknown" for record in records),
        recovery=tuple(decision.to_public_dict() for decision in manager.last_reconciliation),
        admission=manager._admission.snapshot().to_public_dict(),
        unreaped_workers=manager.unreaped_workers,
        resource_profiles=tuple(
            StudioJobResourceProfile(
                kind=kind,
                default_timeout_seconds=manager._default_timeout_seconds,
                max_artifact_bytes=manager._max_artifact_bytes,
                execution_models=("thread", "process"),
            )
            for kind in allowed_kinds
        ),
    )
