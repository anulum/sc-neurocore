# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Atomic capacity and job creation

"""Admission failures roll back the actual SQLite job and capacity transaction."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from sc_neurocore.studio.platform.jobs import StudioJobContext, StudioJobManager
from sc_neurocore.studio.platform.jobs_ledger import StudioJobLedger
from sc_neurocore.studio.platform.jobs_ledger_creation import create_job
from sc_neurocore.studio.platform.jobs_models import StudioJobRejected
from sc_neurocore.studio.platform.jobs_shared_admission import SharedJobAdmission


def test_observer_cannot_replace_established_root_limits(tmp_path: Path) -> None:
    """Reading another manager's root preserves its limits; conflicting writes fail."""
    owner = StudioJobManager(
        root=tmp_path,
        allowed_kinds=frozenset({"analysis"}),
        default_timeout_seconds=3.0,
        max_concurrent_jobs=1,
        max_queued_jobs=0,
    )
    initial = owner.submit(
        kind="analysis",
        owner="owner",
        request_id=None,
        task=lambda context: {},
        idempotency_key="existing",
    )
    assert owner.wait(initial.job_id, 2.0).status == "completed"
    before = owner._admission.snapshot()
    observer = StudioJobManager(
        root=tmp_path,
        allowed_kinds=frozenset({"analysis"}),
        default_timeout_seconds=3.0,
        max_concurrent_jobs=4,
        max_queued_jobs=9,
    )
    assert observer._admission.snapshot() == before
    assert observer.list_records() == owner.list_records()
    with pytest.raises(StudioJobRejected, match="different admission limits"):
        observer.submit(kind="analysis", owner="owner", request_id=None, task=lambda context: {})
    assert owner._admission.snapshot() == before
    assert len(owner.list_records()) == 1
    duplicate = observer.submit(
        kind="analysis",
        owner="owner",
        request_id=None,
        task=lambda context: {},
        idempotency_key="existing",
    )
    assert duplicate.job_id == initial.job_id
    assert owner._admission.snapshot() == before


@pytest.mark.parametrize("established", [False, True])
def test_transition_insert_failure_rolls_back_job_slot_and_counter(
    tmp_path: Path, established: bool
) -> None:
    """A real database refusal after job insertion leaves no partial admission."""
    manager = StudioJobManager(
        root=tmp_path,
        allowed_kinds=frozenset({"analysis"}),
        default_timeout_seconds=3.0,
        max_concurrent_jobs=1,
        max_queued_jobs=0,
    )
    executions: list[str] = []

    def task(context: StudioJobContext) -> dict[str, object]:
        executions.append(context.job_id)
        return {"value": 7}

    if established:
        initial = manager.submit(kind="analysis", owner="owner", request_id=None, task=task)
        assert manager.wait(initial.job_id, 2.0).status == "completed"
    ledger = manager._ledger
    tables = ("jobs", "job_transitions", "admission_reservations", "admission_config")
    before = {
        table: [tuple(row) for row in ledger.connection().execute(f"SELECT * FROM {table}")]
        for table in tables
    }
    with ledger.transaction() as connection:
        connection.execute(
            "CREATE TRIGGER refuse_admission BEFORE INSERT ON job_transitions "
            "WHEN NEW.sequence=0 BEGIN SELECT RAISE(ABORT,'injected transition refusal'); END"
        )
    with pytest.raises(sqlite3.IntegrityError, match="injected transition refusal"):
        manager.submit(
            kind="analysis",
            owner="owner",
            request_id=None,
            task=task,
            idempotency_key="retry-atomic-request",
        )
    for table in tables:
        assert [
            tuple(row) for row in ledger.connection().execute(f"SELECT * FROM {table}")
        ] == before[table]
    assert len(executions) == int(established)
    assert manager._admission.snapshot().running == 0
    with ledger.transaction() as connection:
        connection.execute("DROP TRIGGER refuse_admission")
    retry = manager.submit(
        kind="analysis",
        owner="owner",
        request_id=None,
        task=task,
        idempotency_key="retry-atomic-request",
    )
    assert manager.wait(retry.job_id, 2.0).status == "completed"
    duplicate = manager.submit(
        kind="analysis",
        owner="owner",
        request_id=None,
        task=task,
        idempotency_key="retry-atomic-request",
    )
    assert duplicate.job_id == retry.job_id
    assert len(executions) == int(established) + 1
    assert manager._admission.snapshot().admitted == int(established) + 1


@pytest.mark.parametrize("foreign", [False, True])
def test_creation_refuses_nonowning_or_inactive_connection(tmp_path: Path, foreign: bool) -> None:
    """An external transaction cannot silently bypass the ledger's atomic owner."""
    ledger = StudioJobLedger(root=tmp_path)
    connection = sqlite3.connect(":memory:") if foreign else ledger.connection()
    if foreign:
        connection.execute("BEGIN")
    try:
        with pytest.raises(ValueError, match="this ledger's active transaction"):
            create_job(
                ledger,
                job_id="sj_0000000000000001",
                kind="analysis",
                actor="owner",
                workspace="default",
                request_id=None,
                idempotency_key=None,
                experiment_sha256=None,
                admission=None,
                execution_model="thread",
                connection=connection,
            )
        assert ledger.list_records() == ()
        assert (
            ledger.connection().execute("SELECT COUNT(*) FROM job_transitions").fetchone()[0] == 0
        )
        assert connection.in_transaction is foreign
    finally:
        if foreign:
            connection.rollback()
            connection.close()
        ledger.close()


@pytest.mark.parametrize("concurrent,queued", [(0, 0), (-1, 0), (1, -1)])
def test_invalid_shared_limits_preserve_the_ledger(
    tmp_path: Path, concurrent: int, queued: int
) -> None:
    """Invalid capacity cannot establish root configuration or reserve work."""
    ledger = StudioJobLedger(root=tmp_path)
    try:
        with pytest.raises(ValueError, match="positive/non-negative"):
            SharedJobAdmission(ledger, max_concurrent=concurrent, max_queued=queued)
        assert ledger.list_records() == ()
        for table in ("admission_config", "admission_reservations"):
            assert ledger.connection().execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0
    finally:
        ledger.close()


@pytest.mark.parametrize(
    "job_id,timeout",
    [("", None), ("sj_valid", -1.0), ("sj_valid", float("nan")), ("sj_valid", float("inf"))],
)
def test_invalid_admission_request_has_no_persisted_side_effects(
    tmp_path: Path, job_id: str, timeout: float | None
) -> None:
    """Malformed admission is refused before any durable job, counter or slot."""
    ledger = StudioJobLedger(root=tmp_path)
    admission = SharedJobAdmission(ledger, max_concurrent=1, max_queued=0)
    try:
        before = admission.snapshot()
        with pytest.raises(ValueError, match="identifier|finite and non-negative"):
            admission.admit(
                job_id=job_id,
                kind="analysis",
                actor="owner",
                workspace="default",
                request_id=None,
                idempotency_key=None,
                experiment_sha256=None,
                admission=None,
                execution_model="thread",
                timeout_seconds=timeout,
            )
        assert ledger.list_records() == ()
        assert admission.snapshot() == before
        for table in ("admission_config", "admission_reservations", "job_transitions"):
            assert ledger.connection().execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0
    finally:
        ledger.close()


def test_occupied_identifier_preserves_foreign_reservation(tmp_path: Path) -> None:
    """An admission naming an occupied ID cannot steal a slot; a fresh submission runs.

    Delegated admission receives the job ID from its caller, so an occupied ID
    reaches the shared admission through its public entry point; a generated
    ID colliding by chance has the same outcome.
    """
    manager = StudioJobManager(
        root=tmp_path,
        allowed_kinds=frozenset({"analysis"}),
        default_timeout_seconds=3.0,
        max_concurrent_jobs=2,
        max_queued_jobs=0,
    )
    occupied = "sj_0123456789abcdef"
    executions: list[str] = []

    def task(context: StudioJobContext) -> dict[str, object]:
        executions.append(context.job_id)
        return {"accepted": True}

    with manager._ledger.transaction() as connection:
        connection.execute(
            "INSERT INTO admission_reservations(job_id,supervisor,state) VALUES(?,?,'running')",
            (occupied, "foreign-supervisor"),
        )
    before = manager.status().admission
    reservation = tuple(
        manager._ledger.connection().execute("SELECT * FROM admission_reservations").fetchone()
    )
    with pytest.raises(StudioJobRejected, match="identifier is already occupied"):
        manager._admission.admit(
            job_id=occupied,
            kind="analysis",
            actor="owner",
            workspace="default",
            request_id=None,
            idempotency_key=None,
            experiment_sha256=None,
            admission=None,
            execution_model="thread",
        )
    assert executions == []
    assert manager.list_records() == ()
    assert manager.status().admission == before
    assert (
        tuple(
            manager._ledger.connection().execute("SELECT * FROM admission_reservations").fetchone()
        )
        == reservation
    )
    assert not (tmp_path / occupied).exists()
    retry = manager.submit(kind="analysis", owner="owner", request_id=None, task=task)
    assert manager.wait(retry.job_id, 2.0).status == "completed"
    assert manager._done_events[retry.job_id].wait(1.0)
    assert executions == [retry.job_id]
    assert manager.record(retry.job_id).result == {"accepted": True}
    assert manager.status().admission["running"] == 1
    assert (
        tuple(
            manager._ledger.connection().execute("SELECT * FROM admission_reservations").fetchone()
        )
        == reservation
    )
