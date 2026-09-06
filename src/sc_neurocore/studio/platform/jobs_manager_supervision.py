# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio job manager supervision callbacks

"""The manager's side of the supervision protocol.

The thread and process supervisors call back into their manager to seed a
sandbox, run the work, commit a state change and read the clock. Those
callbacks are one responsibility — being supervised — and they live apart from
the public surface that starts jobs and the custody surface that reads them.
"""

from __future__ import annotations

import threading
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from sc_neurocore.studio.platform.jobs_manager_access import _commit_supervised_update
from sc_neurocore.studio.platform.jobs_manager_state import _StudioJobManagerState
from sc_neurocore.studio.platform.jobs_manager_process import _write_seed_inputs
from sc_neurocore.studio.platform.jobs_manager_thread import _run_thread_supervised
from sc_neurocore.studio.platform.jobs_models import (
    STUDIO_SEED_INPUT_DIR,
    UTC,
    StudioJobArtifact,
    StudioJobStatus,
    StudioJobTask,
)
from sc_neurocore.studio.platform.jobs_paths import _resolve_job_directory
from sc_neurocore.studio.platform.jobs_process_protocol import _run_process_supervised

if TYPE_CHECKING:  # pragma: no cover - import cycle only matters to type checkers
    from collections.abc import Mapping


class StudioJobSupervision:
    """The callbacks a Studio job supervisor makes on its manager.

    Each method annotates ``self`` as the manager state it needs; the mixin
    holds no state of its own.
    """

    def _write_seed_inputs(
        self: _StudioJobManagerState,
        work_dir: Path,
        seed_inputs: Mapping[str, bytes] | None,
        *,
        seed_dir: str = STUDIO_SEED_INPUT_DIR,
    ) -> None:
        _write_seed_inputs(self, work_dir, seed_inputs, seed_dir=seed_dir)

    def _run_supervised(
        self: _StudioJobManagerState,
        job_id: str,
        work_dir: Path,
        cancel_event: threading.Event,
        done_event: threading.Event,
        task: StudioJobTask,
        timeout_seconds: float,
    ) -> None:
        _run_thread_supervised(
            self, job_id, work_dir, cancel_event, done_event, task, timeout_seconds
        )

    def _run_process_supervised(  # noqa: PLR0913 - mirrors the supervised worker contract
        self: _StudioJobManagerState,
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
        self: _StudioJobManagerState,
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

    def _timestamp_utc(self: _StudioJobManagerState) -> str:
        timestamp = self._clock().astimezone(UTC).replace(microsecond=0)
        return str(timestamp.isoformat().replace("+00:00", "Z"))

    def _job_work_dir(self: _StudioJobManagerState, job_id: str) -> Path:
        return _resolve_job_directory(
            root=self._root,
            job_id=job_id,
            error_message="Studio job path escapes the job root.",
        )

    @staticmethod
    def _utc_now() -> datetime:
        return datetime.now(UTC)


__all__ = ["StudioJobSupervision"]
