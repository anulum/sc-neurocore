# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — the job methods the Studio API relies on

"""The job methods the Studio API and its training helpers call.

The embedded :class:`StudioJobManager` and the isolated
:class:`IsolatedJobManager` both provide them with the same signatures and
outcome shapes, so routes and helpers depend on this structure rather than
on either storage profile.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from sc_neurocore.studio.platform.jobs_models import (
    StudioJobArtifactPayload,
    StudioJobListSnapshot,
    StudioJobPurgeSnapshot,
    StudioJobRecord,
    StudioJobStatusSnapshot,
    StudioProcessJobPayload,
)


class StudioJobService(Protocol):
    """Job submission, observation, control and custody used by the API."""

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
        training_config: Mapping[str, object] | None = None,
    ) -> StudioJobRecord:
        """Submit one process task and return its record."""

    def wait(self, job_id: str, timeout_seconds: float | None = None) -> StudioJobRecord:
        """Observe one job until terminal or the deadline."""

    def record(self, job_id: str) -> StudioJobRecord:
        """Return one job record."""

    def list_records(
        self, *, actor: str | None = None, workspace: str | None = None
    ) -> tuple[StudioJobRecord, ...]:
        """Return records in creation order, scoped when asked."""

    def list_snapshot(
        self, *, actor: str | None = None, workspace: str | None = None
    ) -> StudioJobListSnapshot:
        """Return a path-free snapshot of every visible job."""

    def status(self) -> StudioJobStatusSnapshot:
        """Return aggregate path-free health."""

    def purge_snapshot(
        self, *, limit: int = 100, after: str | None = None
    ) -> StudioJobPurgeSnapshot:
        """Read one operator purge journal page."""

    def cancel(self, job_id: str) -> StudioJobRecord:
        """Request cooperative cancellation for one job."""

    def read_artifact(self, job_id: str, relative_path: str) -> StudioJobArtifactPayload:
        """Read and verify one declared artefact."""

    def read_live_artifact_bytes(
        self, job_id: str, relative_path: str, *, offset: int, max_bytes: int = 64 * 1024
    ) -> tuple[bytes, int]:
        """Read one bounded slice from a live artefact."""

    def send_control_command(
        self,
        job_id: str,
        *,
        command: Mapping[str, object],
        seed_inputs: Mapping[str, bytes] | None = None,
    ) -> None:
        """Deliver a command and control seeds to a running job."""

    def purge_terminal_record(self, job_id: str) -> StudioJobRecord:
        """Purge one terminal job and its custody."""

    @property
    def unreaped_workers(self) -> tuple[str, ...]:
        """Return jobs whose workers were not confirmed stopped."""


__all__ = ["StudioJobService"]
