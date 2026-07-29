# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio process worker protocol

"""Environment, supervision, and result parsing for Studio process workers."""

from __future__ import annotations

import json
import os

# Process workers receive shell-free local argument vectors.
import subprocess  # nosec B404
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from sc_neurocore.studio.platform.jobs_manager_state import _StudioJobManagerState
from sc_neurocore.studio.platform.jobs_models import (
    StudioJobArtifact,
    StudioJobRejected,
    StudioProcessJobPayload,
)


def _process_worker_environment() -> dict[str, str]:
    """Return an import-stable environment for Studio process workers."""

    environment = dict(os.environ)
    src_path = Path(__file__).resolve().parents[3]
    repo_path = src_path.parent
    required_paths = (str(src_path), str(repo_path))
    existing_pythonpath = environment.get("PYTHONPATH")
    if existing_pythonpath:
        paths = existing_pythonpath.split(os.pathsep)
        missing_paths = [path for path in required_paths if path not in paths]
        if missing_paths:
            environment["PYTHONPATH"] = os.pathsep.join((*missing_paths, existing_pythonpath))
    else:
        environment["PYTHONPATH"] = os.pathsep.join(required_paths)
    return environment


@dataclass(frozen=True, slots=True)
class _ProcessWorkerResult:
    """Validated terminal payload read from one process worker."""

    status: Literal["completed", "failed"]
    result: dict[str, object]
    error: str | None
    artifacts: tuple[StudioJobArtifact, ...]


def _validate_process_task_path(task_path: str) -> None:
    """Validate one ``module:function`` process-task import path."""

    module_path, separator, function_name = task_path.partition(":")
    if separator != ":" or not module_path.strip() or not function_name.strip():
        raise StudioJobRejected("Studio process task path must use module:function form.")
    if any(part == "" or not part.isidentifier() for part in module_path.split(".")):
        raise StudioJobRejected("Studio process task module path is invalid.")
    if not function_name.isidentifier():
        raise StudioJobRejected("Studio process task function name is invalid.")


def _json_payload(payload: StudioProcessJobPayload, error_message: str) -> str:
    """Serialize a mapping or raise the stable job-rejection contract."""

    try:
        return json.dumps(dict(payload), sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise StudioJobRejected(error_message) from exc


def _terminate_process(process: subprocess.Popen[bytes]) -> None:
    """Terminate one worker and fall back to a bounded kill."""

    process.terminate()
    try:
        process.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=1.0)


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


def _run_process_supervised(
    manager: _StudioJobManagerState,
    job_id: str,
    work_dir: Path,
    cancel_event: threading.Event,
    done_event: threading.Event,
    task_path: str,
    payload_path: Path,
    result_path: Path,
    timeout_seconds: float,
) -> None:
    """Supervise one isolated worker process to a terminal record."""

    manager._update(job_id, status="running", started_at_utc=manager._timestamp_utc())
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
        str(manager._max_artifact_bytes),
    ]
    process = subprocess.Popen(command, env=_process_worker_environment())  # nosec B603
    deadline = time.monotonic() + timeout_seconds
    while process.poll() is None:
        if cancel_event.is_set():
            _terminate_process(process)
            manager._update(
                job_id,
                status="cancelled",
                finished_at_utc=manager._timestamp_utc(),
                artifacts=_load_process_artifacts(result_path),
            )
            done_event.set()
            return
        if time.monotonic() >= deadline:
            _terminate_process(process)
            manager._update(
                job_id,
                status="timed_out",
                error="Studio job exceeded its timeout.",
                finished_at_utc=manager._timestamp_utc(),
                artifacts=_load_process_artifacts(result_path),
            )
            done_event.set()
            return
        time.sleep(0.01)
    result = _load_process_result(result_path)
    if process.returncode == 0 and result.status == "completed":
        manager._update(
            job_id,
            status="completed",
            result=result.result,
            finished_at_utc=manager._timestamp_utc(),
            artifacts=result.artifacts,
        )
    else:
        manager._update(
            job_id,
            status="failed",
            error=result.error or f"Studio process worker exited with {process.returncode}.",
            finished_at_utc=manager._timestamp_utc(),
            artifacts=result.artifacts,
        )
    done_event.set()
