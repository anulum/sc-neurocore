# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio job contracts

"""Immutable public contracts for the local Studio job sandbox."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import timezone
from typing import TYPE_CHECKING, Literal, TypeAlias

if TYPE_CHECKING:
    from sc_neurocore.studio.platform.jobs_context import StudioJobContext

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
StudioJobExecutionModel = Literal["thread", "process"]
StudioJobTask = Callable[["StudioJobContext"], dict[str, object]]
StudioProcessJobPayload: TypeAlias = Mapping[str, object]
JsonValue: TypeAlias = "str | int | float | bool | None | list[JsonValue] | dict[str, JsonValue]"
STUDIO_SEED_INPUT_DIR = ".studio_seed"
STUDIO_CONTROL_DIR = ".studio_control"
STUDIO_CONTROL_SEED_DIR = ".studio_control_seed"
STUDIO_CONTROL_COMMAND_FILE = "command.json"
STUDIO_JOB_ID_PATTERN = re.compile(r"\Asj_[0-9a-f]{16}\Z")


class StudioJobRejected(ValueError):
    """Raised when a Studio job request violates the local sandbox policy."""


class StudioJobCancelled(RuntimeError):
    """Raised inside a cooperative Studio job when cancellation is requested."""


class StudioJobArtifactUnavailable(RuntimeError):
    """Raised when a declared Studio job artifact cannot be safely served."""


@dataclass(frozen=True, slots=True)
class StudioJobArtifact:
    """Path-free manifest entry for one Studio job artifact."""

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
    execution_model: StudioJobExecutionModel
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
            "execution_model": self.execution_model,
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
    """Path-free execution limits for one Studio job kind."""

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
    process_count: int
    thread_count: int
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
            "process_count": self.process_count,
            "resource_profiles": [profile.to_public_dict() for profile in self.resource_profiles],
            "schema_version": self.schema_version,
            "thread_count": self.thread_count,
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
