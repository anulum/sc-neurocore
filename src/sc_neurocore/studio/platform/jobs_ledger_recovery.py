# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio job recovery after a restart

"""Decide what happened to the jobs a departed supervisor left behind.

Recovery is deliberately unwilling to guess. A job whose lease belongs to a
supervisor that is still running is left alone. A lease held by a supervisor
this host can prove is gone becomes
``interrupted`` — the job did not finish, its artifacts stay in its manifest,
and nothing is re-run to find out. A supervisor this host cannot probe leaves
the job ``unknown``, which is not terminal and awaits verification.

Nothing here promotes a job to ``completed``. A result that was never committed
is not a result.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from sc_neurocore.studio.platform.jobs_ledger_supervisor import supervisor_is_alive
from sc_neurocore.studio.platform.jobs_ledger_schema import record_from_row
from sc_neurocore.studio.platform.jobs_models import UTC, StudioJobStatus

if TYPE_CHECKING:  # pragma: no cover - import cycle only matters to type checkers
    from sc_neurocore.studio.platform.jobs_ledger import StudioJobLedger


@dataclass(frozen=True, slots=True)
class StudioJobReconciliation:
    """What startup recovery decided about one job it found alive.

    Attributes
    ----------
    job_id : str
        The job examined.
    previous_status : StudioJobStatus
        The status the ledger held before recovery.
    status : StudioJobStatus
        The status recovery assigned, or the previous one when it left the job
        alone because another live supervisor owns it.
    reason : str
        Why, in words a runbook can act on.
    """

    job_id: str
    previous_status: StudioJobStatus
    status: StudioJobStatus
    reason: str

    def to_public_dict(self) -> dict[str, str]:
        """Return a path-free JSON representation of this decision."""
        return {
            "job_id": self.job_id,
            "previous_status": self.previous_status,
            "reason": self.reason,
            "status": self.status,
        }


def _expired(expiry: object, now: datetime) -> bool:
    """Return whether a stored lease expiry has passed."""
    if expiry is None:
        return True
    return datetime.fromisoformat(str(expiry).replace("Z", "+00:00")).astimezone(UTC) <= now


def reconcile_ledger(ledger: StudioJobLedger) -> tuple[StudioJobReconciliation, ...]:
    """Resolve every job left alive by a supervisor that is no longer here.

    Parameters
    ----------
    ledger : StudioJobLedger
        The ledger to recover. A lease belonging to this process is probed like
        any other: constructing another manager or reconciling a live manager
        does not imply process death. The identity includes a process-start
        token, so a reused PID does not inherit an earlier process's jobs.

    Returns
    -------
    tuple of StudioJobReconciliation
        One decision per retained job examined, including those left running.
        Concurrently purged jobs are omitted; concurrent updates are retained.
    """
    rows = ledger.live_rows()
    now = ledger.now()
    outcomes: list[StudioJobReconciliation] = []
    for row in rows:
        job_id = str(row["job_id"])
        expected = record_from_row(row)
        previous = expected.status
        lease_owner = None if row["lease_owner"] is None else str(row["lease_owner"])
        expired = _expired(row["lease_expires_at_utc"], now)
        alive = None if lease_owner is None else supervisor_is_alive(lease_owner)
        if alive is True:
            target = previous
            reason = (
                "the supervisor is still running despite an expired lease"
                if expired
                else "the supervisor holding the lease is still running"
            )
        elif alive is None:
            target = "unknown"
            reason = "the supervisor holding the lease cannot be probed from this host"
            if expired:
                reason += "; lease expiry is not proof that the worker stopped"
        else:
            target = "interrupted"
            reason = (
                "the lease expired without a heartbeat"
                if expired
                else "the supervisor holding the lease is gone"
            )
        try:
            if alive is True or (target == "unknown" and previous == "unknown"):
                current = ledger.record(job_id)
            else:
                current = ledger.transition(
                    job_id,
                    target,
                    actor=ledger.supervisor,
                    reason=reason,
                    finished_at_utc=ledger.timestamp() if target == "interrupted" else None,
                    error=f"Studio job did not finish: {reason}."
                    if target == "interrupted"
                    else None,
                    expected_record=expected,
                )
        except KeyError:
            # Only an absent job raises KeyError here: a concurrent terminal
            # purge is not permission to recreate it.
            continue
        if current.status != target:
            reason = "the job changed during reconciliation; the newer record was retained"
        outcomes.append(
            StudioJobReconciliation(
                job_id=job_id,
                previous_status=previous,
                status=current.status,
                reason=reason,
            )
        )
    return tuple(outcomes)


__all__ = ["StudioJobReconciliation", "reconcile_ledger"]
