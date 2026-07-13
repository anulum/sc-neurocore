# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio job path confinement

"""Realpath-based confinement helpers for Studio job sandboxes."""

from __future__ import annotations

import os
from pathlib import Path

from sc_neurocore.studio.platform.jobs_models import (
    STUDIO_JOB_ID_PATTERN,
    StudioJobArtifact,
)


def _find_artifact(
    artifacts: tuple[StudioJobArtifact, ...],
    relative_path: str,
) -> StudioJobArtifact:
    """Return the manifest artifact with an exact relative-path match."""

    for artifact in artifacts:
        if artifact.relative_path == relative_path:
            return artifact
    raise KeyError(relative_path)


def _normalize_artifact_lookup_path(relative_path: str) -> str:
    """Validate a manifest lookup while preserving its exact text."""

    try:
        _relative_path_candidate(relative_path)
    except ValueError as exc:
        raise KeyError(relative_path) from exc
    return relative_path


def _resolve_job_directory(*, root: Path, job_id: str, error_message: str) -> Path:
    """Resolve one generated job identifier below the configured root."""

    if STUDIO_JOB_ID_PATTERN.fullmatch(job_id) is None:
        raise ValueError(error_message)
    return _resolve_confined_child(root=root, relative_path=job_id, error_message=error_message)


def _resolve_confined_nested_child(
    *,
    root: Path,
    subdirectory: str,
    relative_path: str,
    error_message: str,
) -> Path:
    """Resolve a child below a separately confined reserved directory."""

    confined_root = _resolve_confined_child(
        root=root,
        relative_path=subdirectory,
        error_message=error_message,
    )
    return _resolve_confined_child(
        root=confined_root,
        relative_path=relative_path,
        error_message=error_message,
    )


def _relative_path_candidate(
    relative_path: str,
    *,
    error_message: str = "Path must be a confined relative path.",
) -> Path:
    """Return a non-empty traversal-free relative path candidate."""

    candidate = Path(relative_path)
    if (
        candidate.is_absolute()
        or not candidate.parts
        or any(part in ("", ".", "..") for part in candidate.parts)
    ):
        raise ValueError(error_message)
    return candidate


def _is_confined_path(*, root: str, child: str) -> bool:
    """Return whether two real paths share the expected root."""

    try:
        return os.path.commonpath((root, child)) == root
    except ValueError:
        return False


def _resolve_confined_child(*, root: Path, relative_path: str, error_message: str) -> Path:
    """Resolve a relative child and reject symlink or traversal escape."""

    candidate = _relative_path_candidate(relative_path, error_message=error_message)
    resolved_root = os.path.realpath(os.fspath(root))
    resolved = os.path.realpath(os.path.join(resolved_root, os.fspath(candidate)))
    if not _is_confined_path(root=resolved_root, child=resolved):
        raise ValueError(error_message)
    return Path(resolved)
