# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio job manager state protocol

"""Structural state contract shared by Studio job manager helpers."""

from __future__ import annotations

import threading
from _thread import LockType
from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Protocol

from sc_neurocore.studio.platform.jobs_shared_admission import SharedJobAdmission
from sc_neurocore.studio.platform.jobs_ledger import StudioJobLedger
from sc_neurocore.studio.platform.jobs_ledger_recovery import StudioJobReconciliation
from sc_neurocore.studio.platform.jobs_models import (
    StudioJobArtifact,
    StudioJobRecord,
    StudioJobStatus,
    StudioJobTask,
)


class _StudioJobManagerState(Protocol):
    """Private structural interface implemented by ``StudioJobManager``."""

    _root: Path
    _allowed_kinds: frozenset[str]
    _default_timeout_seconds: float
    _max_artifact_bytes: int
    _configured: bool
    _clock: Callable[[], datetime]
    _lock: LockType
    #: The durable record of every job. The dictionaries below hold only the
    #: live handles of jobs this process is supervising right now.
    _ledger: StudioJobLedger
    _default_workspace: str
    _admission: SharedJobAdmission
    _unreaped_workers: set[str]
    _reconciliation: tuple[StudioJobReconciliation, ...]
    _done_events: dict[str, threading.Event]
    _cancel_events: dict[str, threading.Event]

    def _write_seed_inputs(
        self,
        work_dir: Path,
        seed_inputs: Mapping[str, bytes] | None,
        *,
        seed_dir: str,
    ) -> None:
        """Write validated seed payloads into one reserved directory."""

    def _run_supervised(
        self,
        job_id: str,
        work_dir: Path,
        cancel_event: threading.Event,
        done_event: threading.Event,
        task: StudioJobTask,
        timeout_seconds: float,
    ) -> None:
        """Supervise one in-process thread job."""

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
        """Supervise one isolated process job."""

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
        """Replace one immutable job record under the manager lock."""

    def _note_unreaped_worker(self, job_id: str) -> None:
        """Record that one job's worker outlived its terminal state."""

    @property
    def unreaped_workers(self) -> tuple[str, ...]:
        """Return the jobs whose worker was still running when they ended."""

    def _timestamp_utc(self) -> str:
        """Return the manager clock as a stable UTC string."""

    def _job_work_dir(self, job_id: str) -> Path:
        """Resolve one generated job directory below the manager root."""

    def record(
        self, job_id: str, *, actor: str | None = None, workspace: str | None = None
    ) -> StudioJobRecord:
        """Return the latest durable job record, scoped when asked."""

    def list_records(
        self, *, actor: str | None = None, workspace: str | None = None
    ) -> tuple[StudioJobRecord, ...]:
        """Return all records in creation order, scoped when asked."""

    @property
    def last_reconciliation(self) -> tuple[StudioJobReconciliation, ...]:
        """Return the decisions of the most recent recovery pass."""
