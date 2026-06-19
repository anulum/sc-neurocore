# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio local job sandbox

"""Local job sandbox contracts for SC-NeuroCore Studio."""

from __future__ import annotations

import hashlib
import secrets
import threading
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

JOBS_STATUS_SCHEMA_VERSION = "studio.jobs.status.v1"
UTC = timezone.utc

StudioJobStatus = Literal[
    "pending",
    "running",
    "completed",
    "failed",
    "cancelling",
    "cancelled",
    "timed_out",
]
StudioJobTask = Callable[["StudioJobContext"], dict[str, object]]


class StudioJobRejected(ValueError):
    """Raised when a Studio job request violates the local sandbox policy."""


class StudioJobCancelled(RuntimeError):
    """Raised inside a cooperative Studio job when cancellation is requested."""


@dataclass(frozen=True, slots=True)
class StudioJobArtifact:
    """Path-free manifest entry for one Studio job artifact.

    Parameters
    ----------
    relative_path:
        Artifact path relative to the job directory.
    size_bytes:
        Number of bytes written to the artifact.
    sha256:
        SHA-256 digest of the artifact payload.
    """

    relative_path: str
    size_bytes: int
    sha256: str

    def to_public_dict(self) -> dict[str, int | str]:
        """Return a path-free JSON representation of this artifact."""

        return {
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True, slots=True)
class StudioJobRecord:
    """Immutable public state for one local Studio job."""

    job_id: str
    kind: str
    owner: str
    request_id: str | None
    status: StudioJobStatus
    created_at_utc: str
    started_at_utc: str | None = None
    finished_at_utc: str | None = None
    error: str | None = None
    result: dict[str, object] | None = None
    artifacts: tuple[StudioJobArtifact, ...] = field(default_factory=tuple)

    def to_public_dict(self) -> dict[str, object]:
        """Return path-free job state suitable for operator APIs."""

        return {
            "artifacts": [artifact.to_public_dict() for artifact in self.artifacts],
            "created_at_utc": self.created_at_utc,
            "error": self.error,
            "finished_at_utc": self.finished_at_utc,
            "job_id": self.job_id,
            "kind": self.kind,
            "owner": self.owner,
            "request_id": self.request_id,
            "result": self.result,
            "started_at_utc": self.started_at_utc,
            "status": self.status,
        }


@dataclass(frozen=True, slots=True)
class StudioJobStatusSnapshot:
    """Path-free aggregate health for the local Studio job manager."""

    configured: bool
    allowed_kinds: tuple[str, ...]
    active_count: int
    completed_count: int
    failed_count: int
    timed_out_count: int
    schema_version: str = JOBS_STATUS_SCHEMA_VERSION

    def to_public_dict(self) -> dict[str, bool | int | list[str] | str]:
        """Return a JSON-serializable, path-free status snapshot."""

        return {
            "active_count": self.active_count,
            "allowed_kinds": list(self.allowed_kinds),
            "completed_count": self.completed_count,
            "configured": self.configured,
            "failed_count": self.failed_count,
            "schema_version": self.schema_version,
            "timed_out_count": self.timed_out_count,
        }


class StudioJobContext:
    """Execution context passed to one local Studio job task."""

    def __init__(
        self,
        *,
        job_id: str,
        work_dir: Path,
        cancel_event: threading.Event,
    ) -> None:
        self.job_id = job_id
        self._work_dir = work_dir
        self._cancel_event = cancel_event
        self._artifacts: list[StudioJobArtifact] = []

    @property
    def cancelled(self) -> bool:
        """Return whether the manager requested cooperative cancellation."""

        return self._cancel_event.is_set()

    @property
    def artifacts(self) -> tuple[StudioJobArtifact, ...]:
        """Return artifacts written through this context."""

        return tuple(self._artifacts)

    def check_cancelled(self) -> None:
        """Raise when the manager requested cooperative cancellation."""

        if self.cancelled:
            raise StudioJobCancelled("Studio job was cancelled.")

    def write_artifact(self, relative_path: str, payload: bytes | str) -> StudioJobArtifact:
        """Write a confined artifact into the job directory.

        Parameters
        ----------
        relative_path:
            Relative path below the job directory. Absolute paths and traversal
            segments are rejected.
        payload:
            UTF-8 text or bytes to persist.

        Returns
        -------
        StudioJobArtifact
            Path-free manifest entry for the written payload.

        Raises
        ------
        ValueError
            If ``relative_path`` escapes the job directory.
        """

        target_path = self._artifact_path(relative_path)
        data = payload.encode("utf-8") if isinstance(payload, str) else payload
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(data)
        artifact = StudioJobArtifact(
            relative_path=relative_path,
            size_bytes=len(data),
            sha256=hashlib.sha256(data).hexdigest(),
        )
        self._artifacts.append(artifact)
        return artifact

    def _artifact_path(self, relative_path: str) -> Path:
        candidate = Path(relative_path)
        if candidate.is_absolute() or any(part == ".." for part in candidate.parts):
            raise ValueError("Studio job artifact path escapes the job directory.")
        resolved = (self._work_dir / candidate).resolve()
        root = self._work_dir.resolve()
        if root != resolved and root not in resolved.parents:
            raise ValueError("Studio job artifact path escapes the job directory.")
        return resolved


class StudioJobManager:
    """Manage local Studio jobs inside per-job sandbox directories."""

    def __init__(
        self,
        *,
        root: Path,
        allowed_kinds: frozenset[str],
        default_timeout_seconds: float,
        configured: bool = True,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if not allowed_kinds:
            raise ValueError("Studio job manager requires at least one allowed job kind.")
        if default_timeout_seconds <= 0:
            raise ValueError("Studio job timeout must be positive.")
        self._root = root
        self._allowed_kinds = frozenset(sorted(allowed_kinds))
        self._default_timeout_seconds = default_timeout_seconds
        self._configured = configured
        self._clock = clock or self._utc_now
        self._lock = threading.Lock()
        self._records: dict[str, StudioJobRecord] = {}
        self._done_events: dict[str, threading.Event] = {}
        self._cancel_events: dict[str, threading.Event] = {}

    def submit(
        self,
        *,
        kind: str,
        owner: str,
        request_id: str | None,
        task: StudioJobTask,
        timeout_seconds: float | None = None,
    ) -> StudioJobRecord:
        """Submit one local job to the bounded worker supervisor."""

        if kind not in self._allowed_kinds:
            raise StudioJobRejected(f"Studio job kind '{kind}' is not allowed.")
        timeout = self._default_timeout_seconds if timeout_seconds is None else timeout_seconds
        if timeout <= 0:
            raise StudioJobRejected("Studio job timeout must be positive.")
        job_id = f"sj_{secrets.token_hex(8)}"
        work_dir = self._root / job_id
        work_dir.mkdir(parents=True, exist_ok=False)
        cancel_event = threading.Event()
        done_event = threading.Event()
        record = StudioJobRecord(
            job_id=job_id,
            kind=kind,
            owner=owner,
            request_id=request_id,
            status="pending",
            created_at_utc=self._timestamp_utc(),
        )
        with self._lock:
            self._records[job_id] = record
            self._done_events[job_id] = done_event
            self._cancel_events[job_id] = cancel_event
        supervisor = threading.Thread(
            target=self._run_supervised,
            args=(job_id, work_dir, cancel_event, done_event, task, timeout),
            daemon=True,
        )
        supervisor.start()
        return record

    def cancel(self, job_id: str) -> StudioJobRecord:
        """Request cooperative cancellation for one job."""

        with self._lock:
            record = self._records[job_id]
            cancel_event = self._cancel_events[job_id]
            if record.status in ("completed", "failed", "cancelled", "timed_out"):
                return record
            cancel_event.set()
            updated = replace(record, status="cancelling")
            self._records[job_id] = updated
            return updated

    def wait(self, job_id: str, timeout_seconds: float | None = None) -> StudioJobRecord:
        """Wait for one job to finish and return its latest record."""

        with self._lock:
            done_event = self._done_events[job_id]
        done_event.wait(timeout_seconds)
        return self.record(job_id)

    def record(self, job_id: str) -> StudioJobRecord:
        """Return the latest immutable record for one job."""

        with self._lock:
            return self._records[job_id]

    def list_records(self) -> tuple[StudioJobRecord, ...]:
        """Return all known jobs in creation order."""

        with self._lock:
            return tuple(self._records.values())

    def status(self) -> StudioJobStatusSnapshot:
        """Return aggregate, path-free job manager health."""

        records = self.list_records()
        active_statuses = {"pending", "running", "cancelling"}
        return StudioJobStatusSnapshot(
            configured=self._configured,
            allowed_kinds=tuple(sorted(self._allowed_kinds)),
            active_count=sum(record.status in active_statuses for record in records),
            completed_count=sum(record.status == "completed" for record in records),
            failed_count=sum(record.status == "failed" for record in records),
            timed_out_count=sum(record.status == "timed_out" for record in records),
        )

    def _run_supervised(
        self,
        job_id: str,
        work_dir: Path,
        cancel_event: threading.Event,
        done_event: threading.Event,
        task: StudioJobTask,
        timeout_seconds: float,
    ) -> None:
        context = StudioJobContext(
            job_id=job_id,
            work_dir=work_dir,
            cancel_event=cancel_event,
        )
        result_box: dict[str, dict[str, object]] = {}
        error_box: dict[str, BaseException] = {}
        self._update(job_id, status="running", started_at_utc=self._timestamp_utc())

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
            self._update(
                job_id,
                status="timed_out",
                error="Studio job exceeded its timeout.",
                finished_at_utc=self._timestamp_utc(),
                artifacts=context.artifacts,
            )
            done_event.set()
            return
        error = error_box.get("error")
        if isinstance(error, StudioJobCancelled):
            self._update(
                job_id,
                status="cancelled",
                finished_at_utc=self._timestamp_utc(),
                artifacts=context.artifacts,
            )
        elif error is not None:
            self._update(
                job_id,
                status="failed",
                error=str(error),
                finished_at_utc=self._timestamp_utc(),
                artifacts=context.artifacts,
            )
        else:
            self._update(
                job_id,
                status="completed",
                result=result_box.get("result", {}),
                finished_at_utc=self._timestamp_utc(),
                artifacts=context.artifacts,
            )
        done_event.set()

    def _update(
        self,
        job_id: str,
        *,
        status: StudioJobStatus,
        started_at_utc: str | None = None,
        finished_at_utc: str | None = None,
        error: str | None = None,
        result: dict[str, object] | None = None,
        artifacts: tuple[StudioJobArtifact, ...] | None = None,
    ) -> None:
        with self._lock:
            record = self._records[job_id]
            self._records[job_id] = replace(
                record,
                status=status,
                started_at_utc=record.started_at_utc if started_at_utc is None else started_at_utc,
                finished_at_utc=finished_at_utc,
                error=error,
                result=result,
                artifacts=record.artifacts if artifacts is None else artifacts,
            )

    def _timestamp_utc(self) -> str:
        timestamp = self._clock().astimezone(UTC).replace(microsecond=0)
        return timestamp.isoformat().replace("+00:00", "Z")

    @staticmethod
    def _utc_now() -> datetime:
        return datetime.now(UTC)


__all__ = [
    "JOBS_STATUS_SCHEMA_VERSION",
    "StudioJobArtifact",
    "StudioJobCancelled",
    "StudioJobContext",
    "StudioJobManager",
    "StudioJobRecord",
    "StudioJobRejected",
    "StudioJobStatus",
    "StudioJobStatusSnapshot",
    "StudioJobTask",
]
