# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio job creation

"""Create job records and their initial transition under the owning transaction."""

from __future__ import annotations

import json
import sqlite3
from contextlib import nullcontext
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from sc_neurocore.studio.platform.jobs_ledger_schema import StudioJobSubmission, record_from_row
from sc_neurocore.studio.platform.jobs_models import StudioJobExecutionModel

if TYPE_CHECKING:
    from sc_neurocore.studio.platform.jobs_ledger import StudioJobLedger

_INSERT_JOB = """
INSERT INTO jobs (
    job_id, kind, actor, workspace, request_id, idempotency_key,
    experiment_sha256, admission, training_config, execution_model, status,
    created_at_utc, artifacts, lease_owner, lease_expires_at_utc,
    heartbeat_at_utc, sequence
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, '[]', ?, ?, ?, 0)
"""

_INSERT_TRANSITION = """
INSERT INTO job_transitions
    (job_id, sequence, from_status, to_status, at_utc, actor, reason)
VALUES (?, ?, ?, ?, ?, ?, ?)
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
    training_config: Mapping[str, object] | None = None,
    connection: sqlite3.Connection | None = None,
    lease_owner: str | None = None,
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
    training_config : mapping, optional
        Canonical, bounded training configuration stored with admission.
    execution_model : {"thread", "process"}
        How the job is supervised.
    connection : sqlite3.Connection, optional
        This ledger's already active transaction, used by shared admission to
        commit reservation and job together. Other connections are rejected.
    lease_owner : str, optional
        Process generation verified by the storage service for a delegated
        admission. The ordinary in-process path uses this ledger's supervisor.

    Returns
    -------
    StudioJobSubmission
        The stored record and whether it was already there.
    """
    training_config_json: str | None = None
    if training_config is not None:
        if kind != "training":
            raise ValueError("Only a training job can carry a training configuration.")
        from sc_neurocore.studio.training_contract import resolve_training_config

        resolved = resolve_training_config(training_config).to_public_dict()
        training_config_json = json.dumps(resolved, sort_keys=True, separators=(",", ":"))
        if len(training_config_json.encode("utf-8")) > 4096:
            raise ValueError("Training configuration exceeds the 4096-byte admission limit.")
    timestamp = ledger.timestamp()
    if connection is not None and (
        connection is not ledger.connection() or not connection.in_transaction
    ):
        raise ValueError("Job admission requires this ledger's active transaction.")
    if lease_owner is not None and not lease_owner:
        raise ValueError("Job lease owner must be nonempty.")
    with ledger.transaction() if connection is None else nullcontext(connection) as connection:
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
                training_config_json,
                execution_model,
                timestamp,
                ledger.supervisor if lease_owner is None else lease_owner,
                ledger.lease_expiry(),
                timestamp,
            ),
        )
        connection.execute(
            _INSERT_TRANSITION, (job_id, 0, None, "pending", timestamp, actor, "admitted")
        )
        row = connection.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
    return StudioJobSubmission(record_from_row(row), duplicate=False)
