# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio process job submission

"""Submission, seed, and control handling for isolated Studio process jobs."""

from __future__ import annotations

import math
import os
import secrets
import threading
from collections.abc import Mapping
from pathlib import Path

from sc_neurocore.studio.platform.jobs_manager_state import _StudioJobManagerState
from sc_neurocore.studio.platform.jobs_models import (
    STUDIO_CONTROL_COMMAND_FILE,
    STUDIO_CONTROL_COMMAND_MAX_BYTES,
    STUDIO_CONTROL_DIR,
    STUDIO_CONTROL_SEED_DIR,
    STUDIO_SEED_INPUT_DIR,
    StudioJobRecord,
    StudioJobRejected,
    StudioProcessJobPayload,
)
from sc_neurocore.studio.platform.jobs_paths import (
    _resolve_confined_child,
    _resolve_job_directory,
)
from sc_neurocore.studio.platform.jobs_process_protocol import (
    _json_payload,
    _validate_process_task_path,
)


def _submit_process_job(
    manager: _StudioJobManagerState,
    *,
    kind: str,
    owner: str,
    request_id: str | None,
    task_path: str,
    payload: StudioProcessJobPayload,
    timeout_seconds: float | None,
    seed_inputs: Mapping[str, bytes] | None,
    workspace: str | None = None,
    idempotency_key: str | None = None,
    experiment_sha256: str | None = None,
    admission: Mapping[str, object] | None = None,
    training_config: Mapping[str, object] | None = None,
) -> StudioJobRecord:
    """Submit one importable task to an isolated Python process.

    An idempotency key already admitted for this actor and workspace returns
    the earlier job untouched, so no side effect is repeated.
    """

    if kind not in manager._allowed_kinds:
        raise StudioJobRejected(f"Studio job kind '{kind}' is not allowed.")
    timeout = manager._default_timeout_seconds if timeout_seconds is None else timeout_seconds
    if not math.isfinite(timeout) or timeout <= 0:
        raise StudioJobRejected("Studio job timeout must be finite and positive.")
    _validate_process_task_path(task_path)
    payload_json = _json_payload(payload, "Studio process job payload must be JSON.")
    if training_config is not None:
        if kind != "training":
            raise StudioJobRejected("Only training jobs may carry a training configuration.")
        from sc_neurocore.studio.training_contract import resolve_training_config

        resolved_config = resolve_training_config(training_config).to_public_dict()
        payload_config = payload.get("config", payload)
        if payload_config != resolved_config:
            raise StudioJobRejected(
                "Training configuration snapshot does not match the process payload."
            )
    job_id = f"sj_{secrets.token_hex(8)}"
    # A slot first, for the same reason as the thread path: a refused
    # submission never reaches the ledger, and a duplicate holds no slot.
    submission = manager._admission.admit(
        job_id=job_id,
        kind=kind,
        actor=owner,
        workspace=workspace or manager._default_workspace,
        request_id=request_id,
        idempotency_key=idempotency_key,
        experiment_sha256=experiment_sha256,
        admission=admission,
        execution_model="process",
        training_config=training_config,
    )
    if submission.duplicate:
        return submission.record
    try:
        work_dir = _resolve_job_directory(
            root=manager._root,
            job_id=job_id,
            error_message="Studio job path escapes the job root.",
        )
        work_dir.mkdir(parents=True, exist_ok=False)
        manager._write_seed_inputs(work_dir, seed_inputs, seed_dir=STUDIO_SEED_INPUT_DIR)
        payload_path = work_dir / ".studio_process_payload.json"
        result_path = work_dir / ".studio_process_result.json"
        payload_path.write_text(payload_json, encoding="utf-8")
        cancel_event = threading.Event()
        done_event = threading.Event()
        with manager._lock:
            manager._done_events[job_id] = done_event
            manager._cancel_events[job_id] = cancel_event
        supervisor = threading.Thread(
            target=manager._run_process_supervised,
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
    except Exception as exc:
        # ``Thread.start`` raises an ``Exception`` only before a thread exists,
        # so no supervisor owns this job and the slot is released here.
        manager._update(
            job_id,
            status="failed",
            error=f"Studio job could not start: {exc}",
            finished_at_utc=manager._timestamp_utc(),
        )
        with manager._lock:
            done = manager._done_events.get(job_id)
        if done is not None:
            done.set()
        raise
    return submission.record


def _send_process_control_command(
    manager: _StudioJobManagerState,
    job_id: str,
    *,
    command: Mapping[str, object],
    seed_inputs: Mapping[str, bytes] | None,
) -> None:
    """Atomically deliver a command and confined seeds to a running job."""

    record = manager._ledger.record(job_id)
    if record.status != "running":
        raise StudioJobRejected("Studio job is not running.")
    command_json = _json_payload(command, "Studio job control command must be JSON.")
    command_bytes = command_json.encode("utf-8")
    if len(command_bytes) > STUDIO_CONTROL_COMMAND_MAX_BYTES:
        raise StudioJobRejected("Studio job control command exceeds configured size limit.")
    try:
        work_dir = manager._job_work_dir(record.job_id)
    except ValueError as exc:
        raise StudioJobRejected(str(exc)) from exc
    if not work_dir.is_dir():
        raise StudioJobRejected("Studio job work directory is unavailable.")
    manager._write_seed_inputs(work_dir, seed_inputs, seed_dir=STUDIO_CONTROL_SEED_DIR)
    control_dir = _resolve_confined_child(
        root=work_dir,
        relative_path=STUDIO_CONTROL_DIR,
        error_message="Studio job control path escapes the job directory.",
    )
    control_dir.mkdir(parents=True, exist_ok=True)
    command_path = control_dir / STUDIO_CONTROL_COMMAND_FILE
    temp_path = control_dir / f".{STUDIO_CONTROL_COMMAND_FILE}.tmp"
    temp_path.write_bytes(command_bytes)
    os.replace(temp_path, command_path)


def _write_seed_inputs(
    manager: _StudioJobManagerState,
    work_dir: Path,
    seed_inputs: Mapping[str, bytes] | None,
    *,
    seed_dir: str,
) -> None:
    """Write size-bounded binary seeds into one confined reserved directory."""

    if not seed_inputs:
        return
    try:
        seed_root = _resolve_confined_child(
            root=work_dir,
            relative_path=seed_dir,
            error_message="Studio job seed-input path escapes the seed directory.",
        )
    except ValueError as exc:
        raise StudioJobRejected(str(exc)) from exc
    seed_root.mkdir(parents=True, exist_ok=True)
    for relative_path, data in seed_inputs.items():
        if not isinstance(data, bytes | bytearray):
            raise StudioJobRejected("Studio job seed input must be bytes.")
        if len(data) > manager._max_artifact_bytes:
            raise StudioJobRejected("Studio job seed input exceeds configured size limit.")
        try:
            target_path = _resolve_confined_child(
                root=seed_root,
                relative_path=relative_path,
                error_message="Studio job seed-input path escapes the seed directory.",
            )
        except ValueError as exc:
            raise StudioJobRejected(str(exc)) from exc
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(bytes(data))
