# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio job task context

"""Confined artifact, seed, and control I/O for one Studio job task."""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from sc_neurocore.studio.platform.jobs_models import (
    STUDIO_CONTROL_COMMAND_FILE,
    STUDIO_CONTROL_DIR,
    STUDIO_CONTROL_SEED_DIR,
    STUDIO_SEED_INPUT_DIR,
    JsonValue,
    StudioJobArtifact,
    StudioJobArtifactUnavailable,
    StudioJobCancelled,
)
from sc_neurocore.studio.platform.jobs_paths import (
    _resolve_confined_child,
    _resolve_confined_nested_child,
)


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
        """Bind one job identifier to its confined task resources."""

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
        """Write one size-bounded artifact below the job directory."""

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
        """Append one size-bounded JSON event to a confined live log."""

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
        """Validate and declare an existing confined artifact in the manifest."""

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

    def read_seed_input(self, relative_path: str) -> bytes:
        """Read one confined, size-bounded submission seed payload."""

        target_path = _resolve_confined_nested_child(
            root=self._work_dir,
            subdirectory=STUDIO_SEED_INPUT_DIR,
            relative_path=relative_path,
            error_message="Studio job seed-input path escapes the seed directory.",
        )
        if not target_path.is_file():
            raise StudioJobArtifactUnavailable("Studio job seed input is unavailable.")
        data = target_path.read_bytes()
        if len(data) > self._max_artifact_bytes:
            raise ValueError("Studio job seed input exceeds configured size limit.")
        return data

    def poll_control_command(self) -> dict[str, JsonValue] | None:
        """Consume one pending JSON control command exactly once."""

        command_path = self._work_dir / STUDIO_CONTROL_DIR / STUDIO_CONTROL_COMMAND_FILE
        try:
            raw = command_path.read_bytes()
        except FileNotFoundError:
            return None
        command_path.unlink(missing_ok=True)
        try:
            decoded = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("Studio job control command is not valid JSON.") from exc
        if not isinstance(decoded, dict):
            raise ValueError("Studio job control command must be a JSON object.")
        return cast(dict[str, JsonValue], decoded)

    def read_control_seed(self, relative_path: str) -> bytes:
        """Read one confined, size-bounded control seed payload."""

        target_path = _resolve_confined_nested_child(
            root=self._work_dir,
            subdirectory=STUDIO_CONTROL_SEED_DIR,
            relative_path=relative_path,
            error_message="Studio job control-seed path escapes the control-seed directory.",
        )
        if not target_path.is_file():
            raise StudioJobArtifactUnavailable("Studio job control seed is unavailable.")
        data = target_path.read_bytes()
        if len(data) > self._max_artifact_bytes:
            raise ValueError("Studio job control seed exceeds configured size limit.")
        return data

    def _artifact_path(self, relative_path: str) -> Path:
        """Resolve one artifact path below the job directory."""

        return _resolve_confined_child(
            root=self._work_dir,
            relative_path=relative_path,
            error_message="Studio job artifact path escapes the job directory.",
        )
