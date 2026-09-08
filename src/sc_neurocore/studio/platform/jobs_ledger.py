# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio durable job ledger

"""The job records survive the process that made them.

Studio job state used to live in one dictionary. The per-job sandbox
directories and their artifacts outlived the interpreter; the records that gave
them meaning did not. A restarted API answered ``404`` for a job it had
completed a second earlier, a second API process over the same root saw none of
the first one's work, the same request submitted twice ran twice, and a job
that was running when the process died had no terminal state and no way to
acquire one.

This is the durable replacement: one SQLite file under the job root, written in
transactions, with the schema and state machine of
:mod:`sc_neurocore.studio.platform.jobs_ledger_schema` and the recovery rules of
:mod:`sc_neurocore.studio.platform.jobs_ledger_recovery`. Every job carries who
ran it (actor and workspace), what made it unique (idempotency key), what it
ran (effective experiment digest and admission decision), who supervises it now
(lease owner, expiry, heartbeat) and what it produced (terminal result and
artifact manifest).

Single host by design: WAL mode serialises the writers that share one job root.
A distributed worker contract is a separate obligation and is not implied here.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from sc_neurocore.studio.platform.jobs_ledger_recovery import (
    StudioJobReconciliation,
    reconcile_ledger,
)
from sc_neurocore.studio.platform.jobs_ledger_reads import (
    read_live_rows,
    read_record,
    read_records,
    read_transitions,
)
from sc_neurocore.studio.platform.jobs_ledger_schema import (
    LEDGER_FILENAME,
    StudioJobSubmission,
    SCHEMA_V1,
    StudioJobLedgerCorrupt,
    migrate,
)
from sc_neurocore.studio.platform.jobs_ledger_writes import (
    create_job,
    delete_job,
    heartbeat_job,
    transition_job,
)
from sc_neurocore.studio.platform.jobs_ledger_supervisor import supervisor_identity
from sc_neurocore.studio.platform.jobs_models import (
    UTC,
    StudioJobArtifact,
    StudioJobExecutionModel,
    StudioJobRecord,
    StudioJobStatus,
)

DEFAULT_LEASE_SECONDS = 60.0
DEFAULT_BUSY_TIMEOUT_MS = 10_000


class StudioJobLedger:
    """A durable, transactional record of every local Studio job.

    Parameters
    ----------
    root : pathlib.Path
        The job root. The ledger file is created inside it, beside the per-job
        sandbox directories it describes.
    clock : callable, optional
        Returns the current time; defaults to the system UTC clock.
    supervisor : str, optional
        Identity of the supervisor in this process; defaults to
        :func:`~sc_neurocore.studio.platform.jobs_ledger_supervisor.supervisor_identity`.
    lease_seconds : float
        How long a lease stays valid without a heartbeat.
    """

    def __init__(
        self,
        *,
        root: Path,
        clock: Callable[[], datetime] | None = None,
        supervisor: str | None = None,
        lease_seconds: float = DEFAULT_LEASE_SECONDS,
    ) -> None:
        if lease_seconds <= 0:
            raise ValueError("Studio job lease duration must be positive.")
        self._root = root
        self._path = root / LEDGER_FILENAME
        self._clock: Callable[[], datetime] = clock or (lambda: datetime.now(UTC))
        self._supervisor = supervisor or supervisor_identity()
        self._lease_seconds = lease_seconds
        self._local = threading.local()
        root.mkdir(parents=True, exist_ok=True)
        # executescript() commits whatever transaction is open, so the schema is
        # applied on its own. Every statement is idempotent, and the migration
        # that follows runs in a transaction of its own.
        self._connect().executescript(SCHEMA_V1)
        with self.transaction() as connection:
            migrate(connection)

    @property
    def path(self) -> Path:
        """Return the ledger file path."""
        return self._path

    @property
    def supervisor(self) -> str:
        """Return the identity this ledger stamps on leases it takes."""
        return self._supervisor

    def close(self) -> None:
        """Close this thread's connection, if it opened one."""
        connection = getattr(self._local, "connection", None)
        if connection is not None:
            connection.close()
            self._local.connection = None

    def connection(self) -> sqlite3.Connection:
        """Return this thread's connection, opening it on first use."""
        return self._connect()

    def _connect(self) -> sqlite3.Connection:
        connection = getattr(self._local, "connection", None)
        if connection is None:
            connection = sqlite3.connect(
                self._path, timeout=DEFAULT_BUSY_TIMEOUT_MS / 1000.0, isolation_level=None
            )
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute(f"PRAGMA busy_timeout={DEFAULT_BUSY_TIMEOUT_MS}")
            self._local.connection = connection
        return connection

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Run one unit of work; either all of it lands or none of it does."""
        connection = self._connect()
        connection.execute("BEGIN IMMEDIATE")
        try:
            yield connection
        except BaseException:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        if connection.in_transaction:
            connection.execute("COMMIT")

    def now(self) -> datetime:
        """Return the ledger clock, truncated to whole seconds in UTC."""
        return self._clock().astimezone(UTC).replace(microsecond=0)

    def timestamp(self) -> str:
        """Return the ledger clock as a stable UTC string."""
        return self.now().isoformat().replace("+00:00", "Z")

    def lease_expiry(self) -> str:
        """Return when a lease taken now would expire without a heartbeat."""
        expiry = self.now() + timedelta(seconds=self._lease_seconds)
        return expiry.isoformat().replace("+00:00", "Z")

    def create(
        self,
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
        """Admit one job, or return the one that already owns its key."""
        return create_job(
            self,
            job_id=job_id,
            kind=kind,
            actor=actor,
            workspace=workspace,
            request_id=request_id,
            idempotency_key=idempotency_key,
            experiment_sha256=experiment_sha256,
            admission=admission,
            execution_model=execution_model,
        )

    def transition(
        self,
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
    ) -> StudioJobRecord:
        """Move one job to a new status and append the transition.

        When ``expected_record`` is supplied, compare all its public fields
        inside the write transaction first. A changed record is returned
        untouched; an absent job still raises ``KeyError``. This lets recovery
        retain a concurrent result or heartbeat instead of acting on stale data.
        """
        return transition_job(
            self,
            job_id,
            to_status,
            actor=actor,
            reason=reason,
            started_at_utc=started_at_utc,
            finished_at_utc=finished_at_utc,
            error=error,
            result=result,
            artifacts=artifacts,
            expected_record=expected_record,
        )

    def heartbeat(self, job_id: str) -> None:
        """Extend this supervisor's lease on a job it is still running."""
        heartbeat_job(self, job_id)

    def delete(self, job_id: str) -> None:
        """Remove one terminal job and its whole transition history."""
        delete_job(self, job_id)

    def record(
        self, job_id: str, *, actor: str | None = None, workspace: str | None = None
    ) -> StudioJobRecord:
        """Return one job record, scoped to an actor and workspace when given."""
        return read_record(self, job_id, actor=actor, workspace=workspace)

    def list_records(
        self, *, actor: str | None = None, workspace: str | None = None
    ) -> tuple[StudioJobRecord, ...]:
        """Return records in creation order, scoped to an actor and workspace."""
        return read_records(self, actor=actor, workspace=workspace)

    def transitions(self, job_id: str) -> tuple[dict[str, Any], ...]:
        """Return the append-only transition history of one job, in order."""
        return read_transitions(self, job_id)

    def live_rows(self) -> tuple[sqlite3.Row, ...]:
        """Return the stored rows of every job that has not finished."""
        return read_live_rows(self)

    def reconcile(self) -> tuple[StudioJobReconciliation, ...]:
        """Resolve every job left alive by a supervisor that is no longer here."""
        return reconcile_ledger(self)


__all__ = [
    "DEFAULT_BUSY_TIMEOUT_MS",
    "DEFAULT_LEASE_SECONDS",
    "StudioJobLedger",
    "StudioJobLedgerCorrupt",
    "StudioJobReconciliation",
    "StudioJobSubmission",
]
