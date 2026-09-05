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
this host can prove is gone, or one that expired without a heartbeat, becomes
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
        The ledger to recover. Its own supervisor identity is treated as a
        previous incarnation: a live process reconciles at startup, before it
        supervises anything, so a lease already stamped with this identity
        belongs to the process that died.

    Returns
    -------
    tuple of StudioJobReconciliation
        One decision per job examined, including those left running.
    """
    rows = ledger.live_rows()
    now = ledger.now()
    outcomes: list[StudioJobReconciliation] = []
    for row in rows:
        job_id = str(row["job_id"])
        previous: StudioJobStatus = str(row["status"])  # type: ignore[assignment]
        lease_owner = None if row["lease_owner"] is None else str(row["lease_owner"])
        expired = _expired(row["lease_expires_at_utc"], now)
        alive = None if lease_owner is None else supervisor_is_alive(lease_owner)
        if lease_owner == ledger.supervisor:
            alive = False
        if alive is True and not expired:
            outcomes.append(
                StudioJobReconciliation(
                    job_id=job_id,
                    previous_status=previous,
                    status=previous,
                    reason="the supervisor holding the lease is still running",
                )
            )
            continue
        if alive is None and not expired:
            reason = "the supervisor holding the lease cannot be probed from this host"
            if previous != "unknown":
                ledger.transition(job_id, "unknown", actor=ledger.supervisor, reason=reason)
            outcomes.append(
                StudioJobReconciliation(
                    job_id=job_id,
                    previous_status=previous,
                    status="unknown",
                    reason=reason,
                )
            )
            continue
        reason = (
            "the lease expired without a heartbeat"
            if expired
            else "the supervisor holding the lease is gone"
        )
        ledger.transition(
            job_id,
            "interrupted",
            actor=ledger.supervisor,
            finished_at_utc=ledger.timestamp(),
            error=f"Studio job did not finish: {reason}.",
            reason=reason,
        )
        outcomes.append(
            StudioJobReconciliation(
                job_id=job_id,
                previous_status=previous,
                status="interrupted",
                reason=reason,
            )
        )
    return tuple(outcomes)


__all__ = ["StudioJobReconciliation", "reconcile_ledger"]
