# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Transactional shared-root job capacity

"""Reserve job capacity across managers and processes using the job ledger."""

from __future__ import annotations

import math
import sqlite3
import time
from collections.abc import Mapping

from sc_neurocore.studio.platform.jobs_admission import AdmissionSnapshot, StudioJobQueueFull
from sc_neurocore.studio.platform.jobs_admission_recovery import recover_stopped_reservations
from sc_neurocore.studio.platform.jobs_admission_replay import (
    StorageAdmissionReplay,
    read_admission_replay,
    write_admission_replay,
)
from sc_neurocore.studio.platform.jobs_ledger import StudioJobLedger
from sc_neurocore.studio.platform.jobs_ledger_schema import StudioJobSubmission, record_from_row
from sc_neurocore.studio.platform.jobs_ledger_supervisor import supervisor_is_alive
from sc_neurocore.studio.platform.jobs_ledger_writes import create_job
from sc_neurocore.studio.platform.jobs_models import StudioJobRejected, StudioJobExecutionModel


class SharedJobAdmission:
    """One root's capacity, with job-scoped reservations and release.

    Opening an observer does not change configuration. The first submission
    establishes root limits; conflicting submission limits are rejected.
    """

    def __init__(self, ledger: StudioJobLedger, *, max_concurrent: int, max_queued: int) -> None:
        if max_concurrent <= 0 or max_queued < 0:
            raise ValueError("Studio shared admission limits must be positive/non-negative.")
        self._ledger = ledger
        self._max_concurrent = max_concurrent
        self._max_queued = max_queued

    @staticmethod
    def _counts(connection: sqlite3.Connection) -> tuple[int, int]:
        rows = connection.execute(
            "SELECT state,COUNT(*) AS total FROM admission_reservations GROUP BY state"
        ).fetchall()
        counts = {str(row["state"]): int(row["total"]) for row in rows}
        legacy = connection.execute(
            "SELECT COUNT(*) FROM jobs WHERE status IN ('pending','running','cancelling','unknown') "
            "AND job_id NOT IN (SELECT job_id FROM admission_reservations)"
        ).fetchone()[0]
        return counts.get("running", 0) + counts.get("unreaped", 0) + int(legacy), counts.get(
            "queued", 0
        )

    def admit(
        self,
        *,
        job_id: str,
        kind: str,
        actor: str,
        workspace: str,
        request_id: str | None,
        idempotency_key: str | None,
        experiment_sha256: str | None,
        admission: Mapping[str, object] | None,
        execution_model: StudioJobExecutionModel,
        training_config: Mapping[str, object] | None = None,
        timeout_seconds: float | None = None,
        replay: StorageAdmissionReplay | None = None,
        supervisor: str | None = None,
    ) -> StudioJobSubmission:
        """Atomically admit capacity and job, or return the existing scoped key.

        Duplicate lookup precedes capacity rejection inside the same transaction.
        Queued requests recheck the key each cycle; only one job is ever inserted.
        A delegated supervisor must come from the service's verified peer, not
        request content. Liveness is checked again while waiting in the queue.
        """
        if not job_id:
            raise ValueError("A reservation requires a job identifier.")
        if supervisor is not None and not supervisor:
            raise ValueError("A delegated supervisor identity must be nonempty.")
        if supervisor is not None and supervisor_is_alive(supervisor) is not True:
            raise StudioJobRejected("Delegated Studio supervisor is not provably live.")
        owner = self._ledger.supervisor if supervisor is None else supervisor
        if replay is not None:
            replay.validate()
            if idempotency_key is not None:
                raise ValueError("Storage replay cannot use the legacy idempotency key.")
        if timeout_seconds is not None and (
            not math.isfinite(timeout_seconds) or timeout_seconds < 0
        ):
            raise ValueError("Admission timeout must be finite and non-negative.")
        deadline = None if timeout_seconds is None else time.monotonic() + timeout_seconds
        queued = False
        try:
            while True:
                if supervisor is not None and supervisor_is_alive(owner) is not True:
                    raise StudioJobRejected("Delegated Studio supervisor exited during admission.")
                refusal: StudioJobQueueFull | None = None
                with self._ledger.transaction() as connection:
                    if replay is not None:
                        prior_outcome = read_admission_replay(
                            connection, workspace=workspace, replay=replay
                        )
                        if prior_outcome is not None:
                            if queued:
                                connection.execute(
                                    "DELETE FROM admission_reservations WHERE job_id=? AND supervisor=?",
                                    (job_id, owner),
                                )
                            if isinstance(prior_outcome, StudioJobQueueFull):
                                raise prior_outcome
                            return prior_outcome
                    if idempotency_key is not None:
                        prior = connection.execute(
                            "SELECT * FROM jobs WHERE actor=? AND workspace=? AND idempotency_key=?",
                            (actor, workspace, idempotency_key),
                        ).fetchone()
                        if prior is not None:
                            if queued:
                                connection.execute(
                                    "DELETE FROM admission_reservations WHERE job_id=? AND supervisor=?",
                                    (job_id, owner),
                                )
                            # Replay and idempotency keys are exclusive, so no replay here.
                            return StudioJobSubmission(record_from_row(prior), duplicate=True)
                    if supervisor is not None and supervisor_is_alive(owner) is not True:
                        raise StudioJobRejected(
                            "Delegated Studio supervisor exited during admission."
                        )
                    connection.execute(
                        "INSERT OR IGNORE INTO admission_config(singleton,max_concurrent,max_queued) "
                        "VALUES(1,?,?)",
                        (self._max_concurrent, self._max_queued),
                    )
                    config = connection.execute(
                        "SELECT * FROM admission_config WHERE singleton=1"
                    ).fetchone()
                    if (config["max_concurrent"], config["max_queued"]) != (
                        self._max_concurrent,
                        self._max_queued,
                    ):
                        raise StudioJobRejected("Studio job root has different admission limits.")
                    running, waiting = self._counts(connection)
                    existing = connection.execute(
                        "SELECT state,supervisor FROM admission_reservations WHERE job_id=?",
                        (job_id,),
                    ).fetchone()
                    if existing is not None and not queued:
                        raise StudioJobRejected(
                            "Studio reservation identifier is already occupied."
                        )
                    first = connection.execute(
                        "SELECT job_id FROM admission_reservations WHERE state='queued' ORDER BY ticket LIMIT 1"
                    ).fetchone()
                    can_run = running < self._max_concurrent and (
                        first is None or first[0] == job_id
                    )
                    if can_run:
                        if queued:
                            connection.execute(
                                "UPDATE admission_reservations SET state='running' WHERE job_id=?",
                                (job_id,),
                            )
                        else:
                            connection.execute(
                                "INSERT INTO admission_reservations(job_id,supervisor,state) VALUES(?,?,'running')",
                                (job_id, owner),
                            )
                        connection.execute(
                            "UPDATE admission_config SET admitted=admitted+1 WHERE singleton=1"
                        )
                        outcome = create_job(
                            self._ledger,
                            job_id=job_id,
                            kind=kind,
                            actor=actor,
                            workspace=workspace,
                            request_id=request_id,
                            idempotency_key=idempotency_key,
                            experiment_sha256=experiment_sha256,
                            admission=admission,
                            execution_model=execution_model,
                            training_config=training_config,
                            connection=connection,
                            lease_owner=owner,
                        )
                        if replay is not None:
                            write_admission_replay(
                                connection, workspace=workspace, replay=replay, outcome=outcome
                            )
                        return outcome
                    expired = deadline is not None and time.monotonic() >= deadline
                    if expired or (not queued and waiting >= self._max_queued):
                        connection.execute(
                            "UPDATE admission_config SET refused=refused+1 WHERE singleton=1"
                        )
                        refusal = StudioJobQueueFull(
                            running=running, queued=waiting, limit=self._max_queued
                        )
                        if queued:
                            connection.execute(
                                "DELETE FROM admission_reservations WHERE job_id=? AND supervisor=?",
                                (job_id, owner),
                            )
                        if replay is not None:
                            write_admission_replay(
                                connection, workspace=workspace, replay=replay, outcome=refusal
                            )
                    elif not queued:
                        connection.execute(
                            "INSERT INTO admission_reservations(job_id,supervisor,state) VALUES(?,?,'queued')",
                            (job_id, owner),
                        )
                        queued = True
                if refusal is not None:
                    raise refusal
                time.sleep(0.05)
        except BaseException:
            if queued:
                self.release(job_id=job_id, supervisor=owner)
            raise

    def release(self, *, job_id: str, supervisor: str | None = None) -> None:
        """Release this supervisor's exact reservation; repeated release is a no-op."""
        owner = self._ledger.supervisor if supervisor is None else supervisor
        with self._ledger.transaction() as connection:
            connection.execute(
                "DELETE FROM admission_reservations WHERE job_id=? AND supervisor=? AND state!='unreaped'",
                (job_id, owner),
            )

    def reconcile(self) -> tuple[str, ...]:
        """Release only reservations whose stopped work has been proved after job recovery."""
        return recover_stopped_reservations(self._ledger)

    def mark_unreaped(self, *, job_id: str, supervisor: str | None = None) -> None:
        """Keep capacity occupied when a terminal job's worker has not stopped."""
        owner = self._ledger.supervisor if supervisor is None else supervisor
        with self._ledger.transaction() as connection:
            connection.execute(
                "UPDATE admission_reservations SET state='unreaped' WHERE job_id=? AND supervisor=?",
                (job_id, owner),
            )

    def snapshot(self) -> AdmissionSnapshot:
        """Read one consistent root-wide occupancy and cumulative counter snapshot."""
        with self._ledger.transaction() as connection:
            running, queued = self._counts(connection)
            config = connection.execute(
                "SELECT * FROM admission_config WHERE singleton=1"
            ).fetchone()
            return AdmissionSnapshot(
                running=running,
                queued=queued,
                max_concurrent=self._max_concurrent
                if config is None
                else int(config["max_concurrent"]),
                max_queued=self._max_queued if config is None else int(config["max_queued"]),
                admitted=0 if config is None else int(config["admitted"]),
                refused=0 if config is None else int(config["refused"]),
            )
