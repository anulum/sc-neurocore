# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio worker result validation

"""Decode and validate results and artifact manifests from worker processes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from sc_neurocore.studio.platform.jobs_models import StudioJobArtifact


@dataclass(frozen=True, slots=True)
class _ProcessWorkerResult:
    """Validated terminal payload read from one process worker."""

    status: Literal["completed", "failed"]
    result: dict[str, object]
    error: str | None
    artifacts: tuple[StudioJobArtifact, ...]


def _load_process_result(result_path: Path) -> _ProcessWorkerResult:
    """Load one worker result or return a stable path-free failure."""
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
    """Return validated worker artifacts, or an empty tuple when absent."""
    if not result_path.exists():
        return ()
    return _load_process_result(result_path).artifacts


def _parse_process_result(payload: dict[object, object]) -> _ProcessWorkerResult:
    """Narrow an untrusted result mapping to the worker result contract."""
    raw_status = payload.get("status")
    status: Literal["completed", "failed"] = "completed" if raw_status == "completed" else "failed"
    raw_result = payload.get("result")
    result = cast(dict[str, object], raw_result) if isinstance(raw_result, dict) else {}
    raw_error = payload.get("error")
    error = raw_error if isinstance(raw_error, str) else None
    artifacts = _parse_process_artifacts(payload.get("artifacts"))
    return _ProcessWorkerResult(
        status=status,
        result=result,
        error=error,
        artifacts=artifacts,
    )


def _parse_process_artifacts(raw_artifacts: object) -> tuple[StudioJobArtifact, ...]:
    """Validate a worker artifact list without accepting partial manifests."""
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
