# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio job manager

"""Public manager coordinating the focused Studio job helpers."""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path

from sc_neurocore.studio.platform.jobs_ledger import (
    DEFAULT_LEASE_SECONDS,
    StudioJobLedger,
    StudioJobReconciliation,
)
from sc_neurocore.studio.platform.jobs_manager_access import (
    _cancel_job,
    _commit_supervised_update,
    _job_manager_status,
    _wait_for_job,
)
from sc_neurocore.studio.platform.jobs_manager_custody import StudioJobCustody
from sc_neurocore.studio.platform.jobs_manager_process import (
    _send_process_control_command,
    _submit_process_job,
    _write_seed_inputs,
)
from sc_neurocore.studio.platform.jobs_manager_thread import (
    _run_thread_supervised,
    _submit_thread_job,
)
from sc_neurocore.studio.platform.jobs_models import (
    DEFAULT_STUDIO_JOB_MAX_ARTIFACT_BYTES,
    STUDIO_SEED_INPUT_DIR,
    UTC,
    StudioJobArtifact,
    StudioJobRecord,
    StudioJobStatus,
    StudioJobStatusSnapshot,
    StudioJobTask,
    StudioProcessJobPayload,
)
from sc_neurocore.studio.platform.jobs_paths import _resolve_job_directory
from sc_neurocore.studio.platform.jobs_process_protocol import _run_process_supervised


class StudioJobManager(StudioJobCustody):
    """Start and supervise local Studio jobs inside per-job sandbox directories.

    Reading what those jobs did is the custody surface this inherits from
    :class:`~sc_neurocore.studio.platform.jobs_manager_custody.StudioJobCustody`.
    """

    def __init__(
        self,
        *,
        root: Path,
        allowed_kinds: frozenset[str],
        default_timeout_seconds: float,
        max_artifact_bytes: int = DEFAULT_STUDIO_JOB_MAX_ARTIFACT_BYTES,
        configured: bool = True,
        clock: Callable[[], datetime] | None = None,
        workspace: str = "default",
        lease_seconds: float = DEFAULT_LEASE_SECONDS,
        reconcile: bool = True,
    ) -> None:
        """Configure bounded execution over the durable job ledger.

        The ledger file lives in ``root`` beside the per-job sandboxes it
        describes, so a restarted process and a second process over the same
        root see the same jobs. Unless ``reconcile`` is disabled, construction
        resolves every job an earlier supervisor left alive.
        """

        if not allowed_kinds:
            raise ValueError("Studio job manager requires at least one allowed job kind.")
        if default_timeout_seconds <= 0:
            raise ValueError("Studio job timeout must be positive.")
        if max_artifact_bytes <= 0:
            raise ValueError("Studio job artifact size limit must be positive.")
        self._root = root.resolve()
        self._allowed_kinds = frozenset(sorted(allowed_kinds))
        self._default_timeout_seconds = default_timeout_seconds
        self._max_artifact_bytes = max_artifact_bytes
        self._configured = configured
        self._clock = clock or self._utc_now
        self._lock = threading.Lock()
        self._default_workspace = workspace
        self._ledger = StudioJobLedger(
            root=self._root, clock=self._clock, lease_seconds=lease_seconds
        )
        # Live handles for the jobs this process supervises. The durable state
        # is the ledger's; these only let this process wait and cancel.
        self._done_events: dict[str, threading.Event] = {}
        self._cancel_events: dict[str, threading.Event] = {}
        self._reconciliation: tuple[StudioJobReconciliation, ...] = ()
        if reconcile:
            self._reconciliation = self._ledger.reconcile()

    def submit(
        self,
        *,
        kind: str,
        owner: str,
        request_id: str | None,
        task: StudioJobTask,
        timeout_seconds: float | None = None,
        workspace: str | None = None,
        idempotency_key: str | None = None,
        experiment_sha256: str | None = None,
        admission: Mapping[str, object] | None = None,
    ) -> StudioJobRecord:
        """Submit one local task to the bounded thread supervisor.

        The job is admitted to the durable ledger before it starts, so it
        survives this process. See
        :func:`~sc_neurocore.studio.platform.jobs_manager_thread._submit_thread_job`
        for the custody fields (``workspace``, ``idempotency_key``,
        ``experiment_sha256``, ``admission``) and what each one binds.
        """

        return _submit_thread_job(
            self,
            kind=kind,
            owner=owner,
            request_id=request_id,
            task=task,
            timeout_seconds=timeout_seconds,
            workspace=workspace,
            idempotency_key=idempotency_key,
            experiment_sha256=experiment_sha256,
            admission=admission,
        )

    def submit_process_task(
        self,
        *,
        kind: str,
        owner: str,
        request_id: str | None,
        task_path: str,
        payload: StudioProcessJobPayload,
        timeout_seconds: float | None = None,
        seed_inputs: Mapping[str, bytes] | None = None,
        workspace: str | None = None,
        idempotency_key: str | None = None,
        experiment_sha256: str | None = None,
        admission: Mapping[str, object] | None = None,
    ) -> StudioJobRecord:
        """Submit one importable task to an isolated Python process.

        Takes the same custody fields as :meth:`submit`; an idempotency key
        already admitted returns the earlier job without starting a process.
        """

        return _submit_process_job(
            self,
            kind=kind,
            owner=owner,
            request_id=request_id,
            task_path=task_path,
            payload=payload,
            timeout_seconds=timeout_seconds,
            seed_inputs=seed_inputs,
            workspace=workspace,
            idempotency_key=idempotency_key,
            experiment_sha256=experiment_sha256,
            admission=admission,
        )

    def send_control_command(
        self,
        job_id: str,
        *,
        command: Mapping[str, object],
        seed_inputs: Mapping[str, bytes] | None = None,
    ) -> None:
        """Atomically deliver control data to one running process job."""

        _send_process_control_command(
            self,
            job_id,
            command=command,
            seed_inputs=seed_inputs,
        )

    def cancel(self, job_id: str) -> StudioJobRecord:
        """Request cooperative cancellation for one job."""

        return _cancel_job(self, job_id)

    def wait(self, job_id: str, timeout_seconds: float | None = None) -> StudioJobRecord:
        """Wait for one job and return its latest immutable record."""

        return _wait_for_job(self, job_id, timeout_seconds)

    def status(self) -> StudioJobStatusSnapshot:
        """Return aggregate path-free manager health."""

        return _job_manager_status(self)

    def _write_seed_inputs(
        self,
        work_dir: Path,
        seed_inputs: Mapping[str, bytes] | None,
        *,
        seed_dir: str = STUDIO_SEED_INPUT_DIR,
    ) -> None:
        _write_seed_inputs(self, work_dir, seed_inputs, seed_dir=seed_dir)

    def _run_supervised(
        self,
        job_id: str,
        work_dir: Path,
        cancel_event: threading.Event,
        done_event: threading.Event,
        task: StudioJobTask,
        timeout_seconds: float,
    ) -> None:
        _run_thread_supervised(
            self,
            job_id,
            work_dir,
            cancel_event,
            done_event,
            task,
            timeout_seconds,
        )

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
        _run_process_supervised(
            self,
            job_id,
            work_dir,
            cancel_event,
            done_event,
            task_path,
            payload_path,
            result_path,
            timeout_seconds,
        )

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
        _commit_supervised_update(
            self,
            job_id,
            status=status,
            started_at_utc=started_at_utc,
            finished_at_utc=finished_at_utc,
            error=error,
            result=result,
            artifacts=artifacts,
        )

    def _timestamp_utc(self) -> str:
        timestamp = self._clock().astimezone(UTC).replace(microsecond=0)
        return timestamp.isoformat().replace("+00:00", "Z")

    def _job_work_dir(self, job_id: str) -> Path:
        return _resolve_job_directory(
            root=self._root,
            job_id=job_id,
            error_message="Studio job path escapes the job root.",
        )

    @staticmethod
    def _utc_now() -> datetime:
        return datetime.now(UTC)
