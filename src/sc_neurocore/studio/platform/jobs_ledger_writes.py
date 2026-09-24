# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio job ledger writes

"""Everything that changes a job: admission, transitions, heartbeat, purge.

Each function is one transaction. Admission is idempotent, a transition is
refused unless the state machine allows it, and a purge takes the whole history
with it rather than leaving orphaned transitions behind.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import nullcontext
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from sc_neurocore.studio.platform.jobs_ledger_creation import (
    create_job as create_job,
    _INSERT_TRANSITION,
)
from sc_neurocore.studio.platform.jobs_ledger_schema import (
    ALLOWED_TRANSITIONS,
    TERMINAL_STATUSES,
    artifacts_to_json,
    record_from_row,
)
from sc_neurocore.studio.platform.jobs_models import (
    StudioJobArtifact,
    StudioJobRecord,
    StudioJobRejected,
    StudioJobStatus,
)

if TYPE_CHECKING:  # pragma: no cover - import cycle only matters to type checkers
    from sc_neurocore.studio.platform.jobs_ledger import StudioJobLedger

_UPDATE_JOB = """
UPDATE jobs SET
    status = ?,
    started_at_utc = COALESCE(?, started_at_utc),
    finished_at_utc = COALESCE(?, finished_at_utc),
    error = COALESCE(?, error),
    result = COALESCE(?, result),
    artifacts = COALESCE(?, artifacts),
    lease_owner = CASE WHEN ? THEN NULL ELSE lease_owner END,
    lease_expires_at_utc = CASE WHEN ? THEN NULL ELSE ? END,
    heartbeat_at_utc = ?,
    sequence = ?
WHERE job_id = ?
"""


def transition_job(
    ledger: StudioJobLedger,
    job_id: str,
    to_status: StudioJobStatus,
    *,
    actor: str | None = None,
    reason: str | None = None,
    started_at_utc: str | None = None,
    finished_at_utc: str | None = None,
    error: str | None = None,
    result: Mapping[str, Any] | None = None,
    artifacts: Sequence[StudioJobArtifact] | None = None,
    expected_record: StudioJobRecord | None = None,
    supervisor: str | None = None,
    connection: sqlite3.Connection | None = None,
) -> StudioJobRecord:
    """Move one job to a new status and append the transition.

    The move is refused when the state machine does not allow it, so a terminal
    record can never be rewritten and an interrupted job can never be quietly
    completed. Repeating a terminal status is a no-op only when all supplied
    fields match the stored values; a conflicting retry is refused. Repeating
    a live status can still record accompanying fields. A supervisor reporting
    ``running`` for a job that is already ``cancelling`` keeps the cancellation
    visible and records only the start time.

    With ``expected_record``, a differing current record is returned without
    changing it. The comparison includes every public field, serialised as JSON
    to preserve numeric and boolean distinctions, and runs under the write lock.

    ``supervisor`` names the lease owner acting through a storage service that
    verified it; ``None`` means this ledger's own supervisor. Only the owner's
    transition renews the lease.

    An optional connection must be this ledger's active transaction, allowing
    the storage authority to commit a terminal record together with the
    release of its admission reservation.

    Raises
    ------
    ValueError
        ``connection`` is not this ledger's active transaction.
    KeyError
        The job is not in the ledger.
    StudioJobRejected
        The transition is not allowed from the job's current status, or a
        supplied field conflicts with the already sealed terminal record.
    """
    if connection is not None and (
        connection is not ledger.connection() or not connection.in_transaction
    ):
        raise ValueError("Job transition requires this ledger's active transaction.")
    acting = ledger.supervisor if supervisor is None else supervisor
    timestamp = ledger.timestamp()
    with ledger.transaction() if connection is None else nullcontext(connection) as connection:
        row = connection.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        if expected_record is not None:
            observed = record_from_row(row)
            if json.dumps(observed.to_public_dict(), sort_keys=True) != json.dumps(
                expected_record.to_public_dict(), sort_keys=True
            ):
                return observed
        current: StudioJobStatus = str(row["status"])  # type: ignore[assignment]
        # A delayed start report must preserve an earlier cancellation.
        # Decide under the write lock to avoid a start/cancel race.
        if to_status == "running" and current == "cancelling":
            to_status = "cancelling"
        unchanged = to_status == current
        if not unchanged and to_status not in ALLOWED_TRANSITIONS[current]:
            raise StudioJobRejected(
                f"Studio job {job_id} cannot move from '{current}' to '{to_status}'."
            )
        if current in TERMINAL_STATUSES:
            supplied = {
                "started_at_utc": started_at_utc,
                "finished_at_utc": finished_at_utc,
                "error": error,
                "result": None if result is None else json.dumps(dict(result), sort_keys=True),
                "artifacts": None if artifacts is None else artifacts_to_json(artifacts),
            }
            changed = [
                name for name, value in supplied.items() if value is not None and value != row[name]
            ]
            if changed:
                raise StudioJobRejected(
                    f"Studio terminal job {job_id} cannot rewrite fields: {', '.join(changed)}."
                )
            return record_from_row(row)
        sequence = int(row["sequence"]) + (0 if unchanged else 1)
        terminal = to_status in TERMINAL_STATUSES
        owns_lease = row["lease_owner"] == acting
        connection.execute(
            _UPDATE_JOB,
            (
                to_status,
                started_at_utc,
                finished_at_utc,
                error,
                None if result is None else json.dumps(dict(result), sort_keys=True),
                None if artifacts is None else artifacts_to_json(artifacts),
                terminal,
                terminal,
                ledger.lease_expiry() if owns_lease else row["lease_expires_at_utc"],
                timestamp if owns_lease else row["heartbeat_at_utc"],
                sequence,
                job_id,
            ),
        )
        if not unchanged:
            connection.execute(
                _INSERT_TRANSITION,
                (
                    job_id,
                    sequence,
                    current,
                    to_status,
                    timestamp,
                    actor or ledger.supervisor,
                    reason,
                ),
            )
        updated = connection.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
    return record_from_row(updated)


def heartbeat_job(ledger: StudioJobLedger, job_id: str, *, supervisor: str | None = None) -> bool:
    """Extend a live job's lease, checking its status in the write transaction.

    Terminal and absent jobs remain unchanged. The transaction serialises this
    check with transitions so a finished job cannot acquire another lease.
    Only the recorded owner can renew a live lease, even after expiry; another
    supervisor raises ``StudioJobRejected`` without changing any stored fields.
    ``supervisor`` is a storage-verified delegated owner; ``None`` means this
    ledger's own supervisor.

    Returns
    -------
    bool
        ``True`` when the lease was renewed, ``False`` when the job is absent
        or already terminal.
    """
    acting = ledger.supervisor if supervisor is None else supervisor
    with ledger.transaction() as connection:
        row = connection.execute(
            "SELECT status, lease_owner FROM jobs WHERE job_id = ?", (job_id,)
        ).fetchone()
        if row is None or row["status"] in TERMINAL_STATUSES:
            return False
        if row["lease_owner"] != acting:
            raise StudioJobRejected(f"Studio job {job_id} lease belongs to another supervisor.")
        connection.execute(
            "UPDATE jobs SET heartbeat_at_utc = ?, lease_expires_at_utc = ?, lease_owner = ?"
            " WHERE job_id = ?",
            (ledger.timestamp(), ledger.lease_expiry(), acting, job_id),
        )
    return True


def require_purgeable(
    connection: sqlite3.Connection, job_id: str, *, purge_supervisor: str | None = None
) -> None:
    """Refuse missing, active or capacity-retaining jobs before discarding custody."""
    row = connection.execute("SELECT status FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
    if row is None:
        raise KeyError(job_id)
    if str(row["status"]) not in TERMINAL_STATUSES:
        raise StudioJobRejected(f"Studio job {job_id} is not terminal and cannot be purged.")
    intent = connection.execute(
        "SELECT supervisor,state FROM job_purges WHERE job_id=?", (job_id,)
    ).fetchone()
    if intent is not None and (
        intent["supervisor"] != purge_supervisor or intent["state"] != "prepared"
    ):
        raise StudioJobRejected(f"Studio job {job_id} has a pending purge requiring recovery.")
    if (
        connection.execute(
            "SELECT 1 FROM admission_reservations WHERE job_id=?", (job_id,)
        ).fetchone()
        is not None
    ):
        raise StudioJobRejected(
            f"Studio job {job_id} retains worker capacity and cannot be purged."
        )


def delete_job(
    ledger: StudioJobLedger, job_id: str, *, connection: sqlite3.Connection | None = None
) -> None:
    """Remove one unreserved terminal job, worker identity and transition history.

    An optional connection must be this ledger's active transaction, allowing
    the filesystem purge owner to serialize staging with record deletion.

    Raises
    ------
    KeyError
        The job is not in the ledger.
    StudioJobRejected
        The job is active or retains capacity; its custody is not disposable.
    """
    if connection is not None and (
        connection is not ledger.connection() or not connection.in_transaction
    ):
        raise ValueError("Job deletion requires this ledger's active transaction.")
    purge_supervisor = ledger.supervisor if connection is not None else None
    with ledger.transaction() if connection is None else nullcontext(connection) as connection:
        require_purgeable(connection, job_id, purge_supervisor=purge_supervisor)
        connection.execute("DELETE FROM job_workers WHERE job_id = ?", (job_id,))
        connection.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,))
        connection.execute("DELETE FROM job_transitions WHERE job_id = ?", (job_id,))


__all__ = ["create_job", "delete_job", "heartbeat_job", "transition_job"]
