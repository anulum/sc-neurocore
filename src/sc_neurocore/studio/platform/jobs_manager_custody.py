# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio job custody surface

"""What a caller can learn about jobs, and what recovery decided about them.

Reading a job, listing jobs, replaying a job's transitions, recovering the ones
a departed supervisor left behind, and serving their artifacts are all the same
responsibility: telling the truth about work that already happened. Starting
and supervising work is the other one, and it lives in
:mod:`sc_neurocore.studio.platform.jobs_manager`.

Record/list reads can be scoped by actor and workspace; an out-of-scope record
raises ``KeyError``. Purge inspection is global operator evidence and requires
administrator authorisation at the API boundary.
"""

from __future__ import annotations

from pathlib import Path

from sc_neurocore.studio.platform.jobs_ledger import StudioJobReconciliation
from sc_neurocore.studio.platform.jobs_purge_recovery import recover_purges
from sc_neurocore.studio.platform.jobs_manager_state import _StudioJobManagerState
from sc_neurocore.studio.platform.jobs_manager_access import (
    _get_job_record,
    _list_job_records,
    _list_job_snapshot,
    _purge_terminal_job,
    _read_declared_artifact,
    _read_live_artifact,
)
from sc_neurocore.studio.platform.jobs_models import (
    StudioJobArtifactPayload,
    StudioJobListSnapshot,
    StudioJobRecord,
    StudioJobPurgeSnapshot,
)


class StudioJobCustody:
    """The durable read surface of a Studio job manager.

    Every method annotates ``self`` as the manager state it needs. The mixin
    carries no state of its own: it is the reading half of one object, split
    from the supervising half so each file has one responsibility.
    """

    def record(
        self: _StudioJobManagerState,
        job_id: str,
        *,
        actor: str | None = None,
        workspace: str | None = None,
    ) -> StudioJobRecord:
        """Return the durable record for one job, scoped when asked.

        A job outside the given actor or workspace raises ``KeyError`` rather
        than leaking its existence.
        """
        return _get_job_record(self, job_id, actor=actor, workspace=workspace)

    def list_records(
        self: _StudioJobManagerState, *, actor: str | None = None, workspace: str | None = None
    ) -> tuple[StudioJobRecord, ...]:
        """Return durable jobs in creation order, scoped when asked."""
        return _list_job_records(self, actor=actor, workspace=workspace)

    def list_snapshot(
        self: _StudioJobManagerState, *, actor: str | None = None, workspace: str | None = None
    ) -> StudioJobListSnapshot:
        """Return a path-free snapshot of every job visible to the caller."""
        return _list_job_snapshot(self, actor=actor, workspace=workspace)

    def purge_snapshot(
        self: _StudioJobManagerState, *, limit: int = 100, after: str | None = None
    ) -> StudioJobPurgeSnapshot:
        """Read a bounded global operator journal page without initiating recovery.

        Validate 1–1000 items and an optional lexical job-ID cursor. This local
        facade does not authenticate callers; HTTP applies the existing ADMIN
        policy. A returned phase is recorded evidence, not a new safety decision.
        """
        return self._ledger.purge_snapshot(limit=limit, after=after)

    def transitions(self: _StudioJobManagerState, job_id: str) -> tuple[dict[str, object], ...]:
        """Return the append-only transition history of one job."""
        return self._ledger.transitions(job_id)

    def reconcile(self: _StudioJobManagerState) -> tuple[StudioJobReconciliation, ...]:
        """Recover dead supervisors and retry this supervisor's committed purge cleanup.

        Live foreign purge intents and this supervisor's prepared operations remain
        untouched. Cleanup failures propagate; ``status().pending_purge_count``
        reports retained intents. The return value describes job reconciliation,
        not purge completion, and no job is re-executed.
        """
        recover_purges(self)
        self._reconciliation = self._ledger.reconcile()
        self._admission.reconcile()
        return self._reconciliation

    @property
    def last_reconciliation(self: _StudioJobManagerState) -> tuple[StudioJobReconciliation, ...]:
        """Return the decisions of the most recent recovery pass."""
        return self._reconciliation

    @property
    def ledger_path(self: _StudioJobManagerState) -> Path:
        """Return the durable ledger file backing this manager."""
        return self._ledger.path

    def purge_terminal_record(self: _StudioJobManagerState, job_id: str) -> StudioJobRecord:
        """Delete one terminal job directory and its in-memory state."""
        return _purge_terminal_job(self, job_id)

    def read_artifact(
        self: _StudioJobManagerState, job_id: str, relative_path: str
    ) -> StudioJobArtifactPayload:
        """Read and verify one manifest-declared artifact."""
        return _read_declared_artifact(self, job_id, relative_path)

    def read_live_artifact_bytes(
        self: _StudioJobManagerState,
        job_id: str,
        relative_path: str,
        *,
        offset: int,
        max_bytes: int = 64 * 1024,
    ) -> tuple[bytes, int]:
        """Read one bounded slice from a confined live artifact."""
        return _read_live_artifact(
            self,
            job_id,
            relative_path,
            offset=offset,
            max_bytes=max_bytes,
        )

    def _note_unreaped_worker(self: _StudioJobManagerState, job_id: str) -> None:
        """Record that one job's worker outlived its terminal state.

        A Python thread cannot be killed, so a task that never checks for
        cancellation keeps running after its record is terminal. That is
        reported rather than hidden.
        """
        with self._lock:
            self._unreaped_workers.add(job_id)

    @property
    def unreaped_workers(self: _StudioJobManagerState) -> tuple[str, ...]:
        """Return the jobs whose worker was still running when they ended."""
        with self._lock:
            return tuple(sorted(self._unreaped_workers))


__all__ = ["StudioJobCustody"]
