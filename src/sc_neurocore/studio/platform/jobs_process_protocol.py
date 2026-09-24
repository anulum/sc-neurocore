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
import sqlite3

# Process workers receive shell-free local argument vectors.
import subprocess  # nosec B404
import sys
import threading
import time
from pathlib import Path

from sc_neurocore.studio.platform.jobs_manager_state import _StudioJobManagerState
from sc_neurocore.studio.platform.jobs_worker_registration import start_worker_registration
from sc_neurocore.studio.platform.jobs_ledger_schema import StudioJobLedgerCorrupt
from sc_neurocore.studio.platform.jobs_models import (
    StudioJobRejected,
    StudioProcessJobPayload,
)
from sc_neurocore.studio.platform.jobs_process_results import (
    _ProcessWorkerResult as _ProcessWorkerResult,
    _load_process_result as _load_process_result,
    _load_process_artifacts as _load_process_artifacts,
    _parse_process_result as _parse_process_result,
    _parse_process_artifacts as _parse_process_artifacts,
)
from sc_neurocore.studio.platform.jobs_reaper import (
    DEFAULT_KILL_GRACE_SECONDS,
    DEFAULT_TERMINATE_GRACE_SECONDS,
    ReapReport,
    _terminate_direct_child,
    reap_process_group,
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
    """Stop one worker that does not lead its own process group.

    Retained for callers that hold a bare child. Supervised jobs go through
    :func:`~sc_neurocore.studio.platform.jobs_reaper.reap_process_group`, which
    stops the worker's descendants too. Both share one implementation, so the
    second wait can no longer raise out of a supervisor that is cleaning up.
    """
    _terminate_direct_child(
        process,
        DEFAULT_TERMINATE_GRACE_SECONDS,
        DEFAULT_KILL_GRACE_SECONDS,
    )


def _unreaped_error(report: ReapReport) -> str:
    """Describe a worker group that survived its reap, so nobody assumes it did not."""
    return (
        f"The worker process group was not reaped after "
        f"{report.duration_seconds:.1f}s; {len(report.survivors)} process(es) may still be running."
    )


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
        "--supervisor",
        manager._ledger.supervisor,
    ]
    # Its own session, so stopping the job stops everything the worker
    # started rather than only the process the supervisor can see.
    process: subprocess.Popen[bytes] | None = None
    try:
        process = subprocess.Popen(  # nosec B603
            command,
            env=_process_worker_environment(),
            start_new_session=True,
            stdin=subprocess.PIPE,
        )
        start_worker_registration(manager._ledger, job_id, process, manager._ledger.supervisor)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        cleanup = ""
        if process is not None:
            report = reap_process_group(process, owned_group_id=process.pid)
            if not report.reaped:
                manager._note_unreaped_worker(job_id)
                cleanup = f" {_unreaped_error(report)}"
        try:
            manager._update(
                job_id,
                status="failed",
                error=f"Studio worker could not start: {exc}{cleanup}",
                finished_at_utc=manager._timestamp_utc(),
            )
        finally:
            done_event.set()
        return
    deadline = time.monotonic() + timeout_seconds
    while process.poll() is None:
        try:
            observed = manager._ledger.record(job_id)
        except (sqlite3.Error, KeyError, StudioJobLedgerCorrupt) as exc:
            report = reap_process_group(process, owned_group_id=process.pid)
            if not report.reaped:
                manager._note_unreaped_worker(job_id)
            try:
                manager._update(
                    job_id,
                    status="failed",
                    error=(
                        f"Studio cancellation observation failed: {exc}. "
                        + ("Worker reaped." if report.reaped else _unreaped_error(report))
                    ),
                    finished_at_utc=manager._timestamp_utc(),
                    artifacts=_load_process_artifacts(result_path),
                )
            finally:
                done_event.set()
            return
        if observed.status == "cancelling":
            cancel_event.set()
        if cancel_event.is_set():
            report = reap_process_group(process, owned_group_id=process.pid)
            if not report.reaped:
                manager._note_unreaped_worker(job_id)
            manager._update(
                job_id,
                status="cancelled",
                error=None if report.reaped else _unreaped_error(report),
                finished_at_utc=manager._timestamp_utc(),
                artifacts=_load_process_artifacts(result_path),
            )
            done_event.set()
            return
        if time.monotonic() >= deadline:
            report = reap_process_group(process, owned_group_id=process.pid)
            if not report.reaped:
                manager._note_unreaped_worker(job_id)
            manager._update(
                job_id,
                status="timed_out",
                error=(
                    "Studio job exceeded its timeout."
                    if report.reaped
                    else f"Studio job exceeded its timeout. {_unreaped_error(report)}"
                ),
                finished_at_utc=manager._timestamp_utc(),
                artifacts=_load_process_artifacts(result_path),
            )
            done_event.set()
            return
        time.sleep(0.01)
    # The worker led a new session. Its descendants retain this group even
    # after poll() collects the direct child and getpgid(pid) stops working.
    report = reap_process_group(process, owned_group_id=process.pid)
    if not report.reaped:
        manager._note_unreaped_worker(job_id)
        try:
            manager._update(
                job_id,
                status="failed",
                error=_unreaped_error(report),
                finished_at_utc=manager._timestamp_utc(),
                artifacts=_load_process_artifacts(result_path),
            )
        finally:
            done_event.set()
        return
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
