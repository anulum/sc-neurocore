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
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from sc_neurocore.studio.platform.jobs_ledger_schema import (
    ALLOWED_TRANSITIONS,
    TERMINAL_STATUSES,
    StudioJobSubmission,
    artifacts_to_json,
    record_from_row,
)
from sc_neurocore.studio.platform.jobs_models import (
    StudioJobArtifact,
    StudioJobExecutionModel,
    StudioJobRecord,
    StudioJobRejected,
    StudioJobStatus,
)

if TYPE_CHECKING:  # pragma: no cover - import cycle only matters to type checkers
    from sc_neurocore.studio.platform.jobs_ledger import StudioJobLedger

_INSERT_JOB = """
INSERT INTO jobs (
    job_id, kind, actor, workspace, request_id, idempotency_key,
    experiment_sha256, admission, execution_model, status,
    created_at_utc, artifacts, lease_owner, lease_expires_at_utc,
    heartbeat_at_utc, sequence
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, '[]', ?, ?, ?, 0)
"""

_INSERT_TRANSITION = """
INSERT INTO job_transitions
    (job_id, sequence, from_status, to_status, at_utc, actor, reason)
VALUES (?, ?, ?, ?, ?, ?, ?)
"""

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


def create_job(
    ledger: StudioJobLedger,
    *,
    job_id: str,
    kind: str,
    actor: str,
    workspace: str,
    request_id: str | None,
    idempotency_key: str | None,
    experiment_sha256: str | None,
    admission: Mapping[str, Any] | None,
    execution_model: StudioJobExecutionModel,
) -> StudioJobSubmission:
    """Admit one job, or return the one that already owns its key.

    Parameters
    ----------
    ledger : StudioJobLedger
        The ledger to write to.
    job_id : str
        Generated identifier for the new job.
    kind, actor, workspace : str
        What is running, for whom, and in which workspace. Actor and workspace
        scope every later read.
    request_id : str, optional
        The caller's request correlation id.
    idempotency_key : str, optional
        When given, a second submission with the same key by the same actor and
        workspace returns the first job instead of starting another.
    experiment_sha256 : str, optional
        Digest of the effective experiment this job runs, when it has one.
    admission : mapping, optional
        The admission decision recorded with the job.
    execution_model : {"thread", "process"}
        How the job is supervised.

    Returns
    -------
    StudioJobSubmission
        The stored record and whether it was already there.
    """
    timestamp = ledger.timestamp()
    with ledger.transaction() as connection:
        if idempotency_key is not None:
            existing = connection.execute(
                "SELECT * FROM jobs WHERE actor = ? AND workspace = ? AND idempotency_key = ?",
                (actor, workspace, idempotency_key),
            ).fetchone()
            if existing is not None:
                return StudioJobSubmission(record_from_row(existing), duplicate=True)
        connection.execute(
            _INSERT_JOB,
            (
                job_id,
                kind,
                actor,
                workspace,
                request_id,
                idempotency_key,
                experiment_sha256,
                json.dumps(dict(admission or {}), sort_keys=True),
                execution_model,
                timestamp,
                ledger.supervisor,
                ledger.lease_expiry(),
                timestamp,
            ),
        )
        connection.execute(
            _INSERT_TRANSITION, (job_id, 0, None, "pending", timestamp, actor, "admitted")
        )
        row = connection.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
    return StudioJobSubmission(record_from_row(row), duplicate=False)


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
) -> StudioJobRecord:
    """Move one job to a new status and append the transition.

    The move is refused when the state machine does not allow it, so a terminal
    record can never be rewritten and an interrupted job can never be quietly
    completed. Repeating the current status is accepted and records the
    accompanying fields without inventing a transition. A supervisor reporting
    ``running`` for a job that is already ``cancelling`` keeps the cancellation
    visible and records only the start time.

    Raises
    ------
    KeyError
        The job is not in the ledger.
    StudioJobRejected
        The transition is not allowed from the job's current status.
    """
    timestamp = ledger.timestamp()
    with ledger.transaction() as connection:
        row = connection.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        current: StudioJobStatus = str(row["status"])  # type: ignore[assignment]
        # A job asked to cancel before its supervisor marked it running stays
        # "cancelling": it did start, and it is already winding down, so
        # reporting it as freshly running would contradict the request the user
        # already made. Deciding this inside the transaction is what makes it
        # race-free; the same check outside would read a status that the cancel
        # changes a moment later.
        if to_status == "running" and current == "cancelling":
            to_status = "cancelling"
        unchanged = to_status == current
        if not unchanged and to_status not in ALLOWED_TRANSITIONS[current]:
            raise StudioJobRejected(
                f"Studio job {job_id} cannot move from '{current}' to '{to_status}'."
            )
        sequence = int(row["sequence"]) + (0 if unchanged else 1)
        terminal = to_status in TERMINAL_STATUSES
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
                ledger.lease_expiry(),
                timestamp,
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


def heartbeat_job(ledger: StudioJobLedger, job_id: str) -> None:
    """Extend a live job's lease, checking its status in the write transaction.

    Terminal and absent jobs remain unchanged. The transaction serialises this
    check with transitions so a finished job cannot acquire another lease.
    """
    with ledger.transaction() as connection:
        row = connection.execute("SELECT status FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        if row is None or row["status"] in TERMINAL_STATUSES:
            return
        connection.execute(
            "UPDATE jobs SET heartbeat_at_utc = ?, lease_expires_at_utc = ?, lease_owner = ?"
            " WHERE job_id = ?",
            (ledger.timestamp(), ledger.lease_expiry(), ledger.supervisor, job_id),
        )


def delete_job(ledger: StudioJobLedger, job_id: str) -> None:
    """Remove one terminal job and its whole transition history.

    Raises
    ------
    KeyError
        The job is not in the ledger.
    StudioJobRejected
        The job has not finished; a running job's history is not disposable.
    """
    with ledger.transaction() as connection:
        row = connection.execute("SELECT status FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        if str(row["status"]) not in TERMINAL_STATUSES:
            raise StudioJobRejected(f"Studio job {job_id} is not terminal and cannot be purged.")
        connection.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,))
        connection.execute("DELETE FROM job_transitions WHERE job_id = ?", (job_id,))


__all__ = ["create_job", "delete_job", "heartbeat_job", "transition_job"]
