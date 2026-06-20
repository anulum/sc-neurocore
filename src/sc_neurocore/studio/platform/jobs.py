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
import json
import secrets
# Process workers use shell-free local argument vectors.
import subprocess  # nosec B404
import sys
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, TypeAlias, cast

JOBS_STATUS_SCHEMA_VERSION = "studio.jobs.status.v1"
JOBS_LIST_SCHEMA_VERSION = "studio.jobs.list.v1"
DEFAULT_STUDIO_JOB_MAX_ARTIFACT_BYTES = 16 * 1024 * 1024
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
StudioProcessJobPayload: TypeAlias = Mapping[str, object]


class StudioJobRejected(ValueError):
    """Raised when a Studio job request violates the local sandbox policy."""


class StudioJobCancelled(RuntimeError):
    """Raised inside a cooperative Studio job when cancellation is requested."""


class StudioJobArtifactUnavailable(RuntimeError):
    """Raised when a declared Studio job artifact cannot be safely served."""


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
class StudioJobResourceProfile:
    """Path-free execution limits for one Studio job kind.

    Parameters
    ----------
    kind:
        Job kind covered by this profile.
    default_timeout_seconds:
        Default wall-clock timeout applied when a job omits an override.
    max_artifact_bytes:
        Maximum size for each artifact written through ``StudioJobContext``.
    execution_models:
        Supported manager execution models for this job kind.
    """

    kind: str
    default_timeout_seconds: float
    max_artifact_bytes: int
    execution_models: tuple[str, ...]

    def to_public_dict(self) -> dict[str, float | int | list[str] | str]:
        """Return a JSON-serializable, path-free resource profile."""

        return {
            "default_timeout_seconds": self.default_timeout_seconds,
            "execution_models": list(self.execution_models),
            "kind": self.kind,
            "max_artifact_bytes": self.max_artifact_bytes,
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
    resource_profiles: tuple[StudioJobResourceProfile, ...]
    schema_version: str = JOBS_STATUS_SCHEMA_VERSION

    def to_public_dict(self) -> dict[str, object]:
        """Return a JSON-serializable, path-free status snapshot."""

        return {
            "active_count": self.active_count,
            "allowed_kinds": list(self.allowed_kinds),
            "completed_count": self.completed_count,
            "configured": self.configured,
            "failed_count": self.failed_count,
            "resource_profiles": [
                profile.to_public_dict() for profile in self.resource_profiles
            ],
            "schema_version": self.schema_version,
            "timed_out_count": self.timed_out_count,
        }


@dataclass(frozen=True, slots=True)
class StudioJobListSnapshot:
    """Path-free list payload for Studio job operator views."""

    records: tuple[StudioJobRecord, ...]
    schema_version: str = JOBS_LIST_SCHEMA_VERSION

    def to_public_dict(self) -> dict[str, object]:
        """Return JSON-serializable job records without filesystem paths."""

        return {
            "jobs": [record.to_public_dict() for record in self.records],
            "schema_version": self.schema_version,
        }


@dataclass(frozen=True, slots=True)
class StudioJobArtifactPayload:
    """Verified payload for one declared Studio job artifact."""

    artifact: StudioJobArtifact
    payload: bytes


class StudioJobContext:
    """Execution context passed to one local Studio job task."""

    def __init__(
        self,
        *,
        job_id: str,
        work_dir: Path,
        cancel_event: threading.Event,
        max_artifact_bytes: int,
    ) -> None:
        self.job_id = job_id
        self._work_dir = work_dir
        self._cancel_event = cancel_event
        self._max_artifact_bytes = max_artifact_bytes
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
        if len(data) > self._max_artifact_bytes:
            raise ValueError("Studio job artifact exceeds configured size limit.")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(data)
        artifact = StudioJobArtifact(
            relative_path=relative_path,
            size_bytes=len(data),
            sha256=hashlib.sha256(data).hexdigest(),
        )
        self._artifacts.append(artifact)
        return artifact

    def append_artifact_event(
        self,
        relative_path: str,
        payload: Mapping[str, object],
    ) -> None:
        """Append one JSON event to a confined live artifact log.

        Parameters
        ----------
        relative_path:
            Relative event-log path below the job directory.
        payload:
            JSON-serializable event object to append as one JSONL row.

        Raises
        ------
        ValueError
            If the path escapes the job directory, the event is not JSON, or
            the event log would exceed the per-artifact byte limit.
        """

        target_path = self._artifact_path(relative_path)
        try:
            line = json.dumps(dict(payload), sort_keys=True) + "\n"
        except (TypeError, ValueError) as exc:
            raise ValueError("Studio job event payload must be JSON.") from exc
        data = line.encode("utf-8")
        current_size = target_path.stat().st_size if target_path.exists() else 0
        if current_size + len(data) > self._max_artifact_bytes:
            raise ValueError("Studio job event log exceeds configured size limit.")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with target_path.open("ab") as handle:
            handle.write(data)

    def publish_existing_artifact(self, relative_path: str) -> StudioJobArtifact:
        """Declare an already-written confined artifact in the manifest.

        Live event logs are appended while a process task runs, then published
        once at terminal state so normal artifact download and integrity checks
        use the same manifest contract as other Studio job outputs.
        """

        target_path = self._artifact_path(relative_path)
        if not target_path.is_file():
            raise ValueError("Studio job artifact is unavailable.")
        data = target_path.read_bytes()
        if len(data) > self._max_artifact_bytes:
            raise ValueError("Studio job artifact exceeds configured size limit.")
        artifact = StudioJobArtifact(
            relative_path=relative_path,
            size_bytes=len(data),
            sha256=hashlib.sha256(data).hexdigest(),
        )
        self._artifacts = [
            existing
            for existing in self._artifacts
            if existing.relative_path != artifact.relative_path
        ]
        self._artifacts.append(artifact)
        return artifact

    def _artifact_path(self, relative_path: str) -> Path:
        return _resolve_confined_child(
            root=self._work_dir,
            relative_path=relative_path,
            error_message="Studio job artifact path escapes the job directory.",
        )


class StudioJobManager:
    """Manage local Studio jobs inside per-job sandbox directories."""

    def __init__(
        self,
        *,
        root: Path,
        allowed_kinds: frozenset[str],
        default_timeout_seconds: float,
        max_artifact_bytes: int = DEFAULT_STUDIO_JOB_MAX_ARTIFACT_BYTES,
        configured: bool = True,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if not allowed_kinds:
            raise ValueError("Studio job manager requires at least one allowed job kind.")
        if default_timeout_seconds <= 0:
            raise ValueError("Studio job timeout must be positive.")
        if max_artifact_bytes <= 0:
            raise ValueError("Studio job artifact size limit must be positive.")
        self._root = root
        self._allowed_kinds = frozenset(sorted(allowed_kinds))
        self._default_timeout_seconds = default_timeout_seconds
        self._max_artifact_bytes = max_artifact_bytes
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

    def submit_process_task(
        self,
        *,
        kind: str,
        owner: str,
        request_id: str | None,
        task_path: str,
        payload: StudioProcessJobPayload,
        timeout_seconds: float | None = None,
    ) -> StudioJobRecord:
        """Submit an importable Studio job task to an isolated Python process.

        Parameters
        ----------
        kind:
            Job kind that must be present in the manager allow-list.
        owner:
            Operator or subsystem label recorded in the path-free job record.
        request_id:
            Optional request correlation identifier.
        task_path:
            Import path in ``module:function`` form. The function must accept
            ``(StudioJobContext, Mapping[str, object])`` and return a
            JSON-serializable ``dict[str, object]``.
        payload:
            JSON-serializable input payload passed to the worker function.
        timeout_seconds:
            Optional per-job timeout. Timed-out process jobs are terminated.

        Returns
        -------
        StudioJobRecord
            Initial pending record for the submitted process job.

        Raises
        ------
        StudioJobRejected
            If the kind, timeout, task import path, or payload is invalid.
        """

        if kind not in self._allowed_kinds:
            raise StudioJobRejected(f"Studio job kind '{kind}' is not allowed.")
        timeout = self._default_timeout_seconds if timeout_seconds is None else timeout_seconds
        if timeout <= 0:
            raise StudioJobRejected("Studio job timeout must be positive.")
        _validate_process_task_path(task_path)
        payload_json = _json_payload(payload, "Studio process job payload must be JSON.")
        job_id = f"sj_{secrets.token_hex(8)}"
        work_dir = self._root / job_id
        work_dir.mkdir(parents=True, exist_ok=False)
        payload_path = work_dir / ".studio_process_payload.json"
        result_path = work_dir / ".studio_process_result.json"
        payload_path.write_text(payload_json, encoding="utf-8")
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
            target=self._run_process_supervised,
            args=(
                job_id,
                work_dir,
                cancel_event,
                done_event,
                task_path,
                payload_path,
                result_path,
                timeout,
            ),
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

    def list_snapshot(self) -> StudioJobListSnapshot:
        """Return a path-free snapshot of all known jobs."""

        return StudioJobListSnapshot(records=self.list_records())

    def read_artifact(self, job_id: str, relative_path: str) -> StudioJobArtifactPayload:
        """Return a verified payload for one declared job artifact.

        The requested path must match a manifest entry exactly, resolve inside
        the owning job directory, and still match the recorded size and SHA-256
        digest. Missing jobs or manifest entries raise ``KeyError`` so callers
        can return a generic not-found response without exposing local paths.
        """

        record = self.record(job_id)
        requested_path = _normalize_artifact_lookup_path(relative_path)
        artifact = _find_artifact(record.artifacts, requested_path)
        artifact_path = _resolve_confined_child(
            root=self._root / job_id,
            relative_path=artifact.relative_path,
            error_message="Studio job artifact path escapes the job directory.",
        )
        if not artifact_path.is_file():
            raise StudioJobArtifactUnavailable("Studio job artifact is unavailable.")
        payload = artifact_path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if len(payload) != artifact.size_bytes or digest != artifact.sha256:
            raise StudioJobArtifactUnavailable("Studio job artifact integrity check failed.")
        return StudioJobArtifactPayload(artifact=artifact, payload=payload)

    def read_live_artifact_bytes(
        self,
        job_id: str,
        relative_path: str,
        *,
        offset: int,
        max_bytes: int = 64 * 1024,
    ) -> tuple[bytes, int]:
        """Return newly appended bytes from a confined live artifact.

        The live reader is intentionally not manifest-gated because child
        processes append evidence rows before terminal artifact publication.
        It still requires a known job, confines the requested relative path to
        that job directory, rejects negative offsets, and returns at most
        ``max_bytes`` bytes per read.
        """

        self.record(job_id)
        if offset < 0:
            raise ValueError("Studio live artifact offset must be non-negative.")
        if max_bytes <= 0:
            raise ValueError("Studio live artifact read size must be positive.")
        requested_path = _normalize_artifact_lookup_path(relative_path)
        artifact_path = _resolve_confined_child(
            root=self._root / job_id,
            relative_path=requested_path,
            error_message="Studio job artifact path escapes the job directory.",
        )
        if not artifact_path.is_file():
            return b"", offset
        with artifact_path.open("rb") as handle:
            handle.seek(offset)
            payload = handle.read(max_bytes)
            return payload, offset + len(payload)

    def status(self) -> StudioJobStatusSnapshot:
        """Return aggregate, path-free job manager health."""

        records = self.list_records()
        active_statuses = {"pending", "running", "cancelling"}
        allowed_kinds = tuple(sorted(self._allowed_kinds))
        return StudioJobStatusSnapshot(
            configured=self._configured,
            allowed_kinds=allowed_kinds,
            active_count=sum(record.status in active_statuses for record in records),
            completed_count=sum(record.status == "completed" for record in records),
            failed_count=sum(record.status == "failed" for record in records),
            timed_out_count=sum(record.status == "timed_out" for record in records),
            resource_profiles=tuple(
                StudioJobResourceProfile(
                    kind=kind,
                    default_timeout_seconds=self._default_timeout_seconds,
                    max_artifact_bytes=self._max_artifact_bytes,
                    execution_models=("thread", "process"),
                )
                for kind in allowed_kinds
            ),
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
            max_artifact_bytes=self._max_artifact_bytes,
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

    def _run_process_supervised(
        self,
        job_id: str,
        work_dir: Path,
        cancel_event: threading.Event,
        done_event: threading.Event,
        task_path: str,
        payload_path: Path,
        result_path: Path,
        timeout_seconds: float,
    ) -> None:
        self._update(job_id, status="running", started_at_utc=self._timestamp_utc())
        command = [
            sys.executable,
            "-m",
            "sc_neurocore.studio.platform.process_worker",
            "--task",
            task_path,
            "--payload",
            str(payload_path),
            "--result",
            str(result_path),
            "--work-dir",
            str(work_dir),
            "--max-artifact-bytes",
            str(self._max_artifact_bytes),
        ]
        process = subprocess.Popen(command)  # nosec B603
        deadline = time.monotonic() + timeout_seconds
        while process.poll() is None:
            if cancel_event.is_set():
                _terminate_process(process)
                self._update(
                    job_id,
                    status="cancelled",
                    finished_at_utc=self._timestamp_utc(),
                    artifacts=_load_process_artifacts(result_path),
                )
                done_event.set()
                return
            if time.monotonic() >= deadline:
                _terminate_process(process)
                self._update(
                    job_id,
                    status="timed_out",
                    error="Studio job exceeded its timeout.",
                    finished_at_utc=self._timestamp_utc(),
                    artifacts=_load_process_artifacts(result_path),
                )
                done_event.set()
                return
            time.sleep(0.01)
        result = _load_process_result(result_path)
        if process.returncode == 0 and result.status == "completed":
            self._update(
                job_id,
                status="completed",
                result=result.result,
                finished_at_utc=self._timestamp_utc(),
                artifacts=result.artifacts,
            )
        else:
            self._update(
                job_id,
                status="failed",
                error=result.error or f"Studio process worker exited with {process.returncode}.",
                finished_at_utc=self._timestamp_utc(),
                artifacts=result.artifacts,
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


def _find_artifact(
    artifacts: tuple[StudioJobArtifact, ...],
    relative_path: str,
) -> StudioJobArtifact:
    for artifact in artifacts:
        if artifact.relative_path == relative_path:
            return artifact
    raise KeyError(relative_path)


def _normalize_artifact_lookup_path(relative_path: str) -> str:
    candidate = Path(relative_path)
    if candidate.is_absolute() or any(part == ".." for part in candidate.parts):
        raise KeyError(relative_path)
    return relative_path


def _resolve_confined_child(*, root: Path, relative_path: str, error_message: str) -> Path:
    candidate = Path(relative_path)
    if candidate.is_absolute() or any(part == ".." for part in candidate.parts):
        raise ValueError(error_message)
    resolved = (root / candidate).resolve()
    resolved_root = root.resolve()
    if resolved_root != resolved and resolved_root not in resolved.parents:
        raise ValueError(error_message)
    return resolved


@dataclass(frozen=True, slots=True)
class _ProcessWorkerResult:
    status: Literal["completed", "failed"]
    result: dict[str, object]
    error: str | None
    artifacts: tuple[StudioJobArtifact, ...]


def _validate_process_task_path(task_path: str) -> None:
    module_path, separator, function_name = task_path.partition(":")
    if separator != ":" or not module_path.strip() or not function_name.strip():
        raise StudioJobRejected("Studio process task path must use module:function form.")
    if any(part == "" or not part.isidentifier() for part in module_path.split(".")):
        raise StudioJobRejected("Studio process task module path is invalid.")
    if not function_name.isidentifier():
        raise StudioJobRejected("Studio process task function name is invalid.")


def _json_payload(payload: StudioProcessJobPayload, error_message: str) -> str:
    try:
        return json.dumps(dict(payload), sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise StudioJobRejected(error_message) from exc


def _terminate_process(process: subprocess.Popen[bytes]) -> None:
    process.terminate()
    try:
        process.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=1.0)


def _load_process_result(result_path: Path) -> _ProcessWorkerResult:
    if not result_path.exists():
        return _ProcessWorkerResult(
            status="failed",
            result={},
            error="Studio process worker did not write a result.",
            artifacts=(),
        )
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _ProcessWorkerResult(
            status="failed",
            result={},
            error="Studio process worker wrote an invalid result.",
            artifacts=(),
        )
    if not isinstance(payload, dict):
        return _ProcessWorkerResult(
            status="failed",
            result={},
            error="Studio process worker wrote an invalid result.",
            artifacts=(),
        )
    return _parse_process_result(payload)


def _load_process_artifacts(result_path: Path) -> tuple[StudioJobArtifact, ...]:
    if not result_path.exists():
        return ()
    return _load_process_result(result_path).artifacts


def _parse_process_result(payload: dict[object, object]) -> _ProcessWorkerResult:
    raw_status = payload.get("status")
    status: Literal["completed", "failed"] = "completed" if raw_status == "completed" else "failed"
    raw_result = payload.get("result")
    result = cast(dict[str, object], raw_result) if isinstance(raw_result, dict) else {}
    raw_error = payload.get("error")
    error = raw_error if isinstance(raw_error, str) else None
    raw_artifacts = payload.get("artifacts")
    artifacts = _parse_process_artifacts(raw_artifacts)
    return _ProcessWorkerResult(
        status=status,
        result=result,
        error=error,
        artifacts=artifacts,
    )


def _parse_process_artifacts(raw_artifacts: object) -> tuple[StudioJobArtifact, ...]:
    if not isinstance(raw_artifacts, list):
        return ()
    artifacts: list[StudioJobArtifact] = []
    for item in raw_artifacts:
        if not isinstance(item, dict):
            return ()
        relative_path = item.get("relative_path")
        size_bytes = item.get("size_bytes")
        sha256 = item.get("sha256")
        if not isinstance(relative_path, str):
            return ()
        if not isinstance(size_bytes, int):
            return ()
        if not isinstance(sha256, str):
            return ()
        artifacts.append(
            StudioJobArtifact(
                relative_path=relative_path,
                size_bytes=size_bytes,
                sha256=sha256,
            )
        )
    return tuple(artifacts)


__all__ = [
    "JOBS_LIST_SCHEMA_VERSION",
    "JOBS_STATUS_SCHEMA_VERSION",
    "DEFAULT_STUDIO_JOB_MAX_ARTIFACT_BYTES",
    "StudioJobArtifact",
    "StudioJobArtifactPayload",
    "StudioJobArtifactUnavailable",
    "StudioJobCancelled",
    "StudioJobContext",
    "StudioJobListSnapshot",
    "StudioJobManager",
    "StudioJobRecord",
    "StudioJobRejected",
    "StudioJobResourceProfile",
    "StudioJobStatus",
    "StudioJobStatusSnapshot",
    "StudioJobTask",
    "StudioProcessJobPayload",
]
