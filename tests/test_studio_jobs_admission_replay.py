# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — durable storage admission replay

"""Exercise exact mutation replay through real shared admission and SQLite."""

from __future__ import annotations

import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import pytest

from sc_neurocore.studio.platform.jobs_admission import StudioJobQueueFull
from sc_neurocore.studio.platform.jobs_admission_replay import StorageAdmissionReplay
from sc_neurocore.studio.platform.jobs_ledger import StudioJobLedger
from sc_neurocore.studio.platform.jobs_ledger_schema import (
    LEDGER_FILENAME,
    SCHEMA_VERSION,
    StudioJobSubmission,
)
from sc_neurocore.studio.platform.jobs_ledger_supervisor import supervisor_identity
from sc_neurocore.studio.platform.jobs_models import StudioJobRejected
from sc_neurocore.studio.platform.jobs_shared_admission import SharedJobAdmission


def _admit(
    admission: SharedJobAdmission,
    *,
    job_id: str,
    replay: StorageAdmissionReplay,
    supervisor: str | None = None,
) -> StudioJobSubmission:
    return admission.admit(
        job_id=job_id,
        kind="analysis",
        actor="studio-service",
        workspace="default",
        request_id="trace-only",
        idempotency_key=None,
        experiment_sha256=None,
        admission={"budget": 3},
        execution_model="process",
        replay=replay,
        supervisor=supervisor,
    )


def test_lost_reply_replays_original_admission_after_restart(tmp_path: Path) -> None:
    """A retry with a fresh job ID does not create or reserve another job."""
    replay = StorageAdmissionReplay("operator", "request-one", "a" * 64)
    first_ledger = StudioJobLedger(root=tmp_path)
    first = SharedJobAdmission(first_ledger, max_concurrent=1, max_queued=0)
    original = _admit(first, job_id="sj_0000000000000001", replay=replay)
    assert not original.duplicate
    first_ledger.close()

    second_ledger = StudioJobLedger(root=tmp_path)
    try:
        second = SharedJobAdmission(second_ledger, max_concurrent=1, max_queued=0)
        duplicate = _admit(second, job_id="sj_0000000000000002", replay=replay)
        assert duplicate.duplicate and duplicate.record == original.record
        assert second.snapshot().admitted == 1
        assert [record.job_id for record in second_ledger.list_records()] == [
            original.record.job_id
        ]
        with pytest.raises(StudioJobRejected, match="changed content"):
            _admit(
                second,
                job_id="sj_0000000000000003",
                replay=StorageAdmissionReplay("operator", "request-one", "b" * 64),
            )
        assert second.snapshot().admitted == 1
    finally:
        second_ledger.close()


def test_delegated_admission_binds_reservation_and_lease_to_api_generation(
    tmp_path: Path,
) -> None:
    """Storage's own process identity must not become the admitted job owner."""
    ledger = StudioJobLedger(root=tmp_path, supervisor="storage:service")
    admission = SharedJobAdmission(ledger, max_concurrent=1, max_queued=0)
    api_generation = supervisor_identity()
    replay = StorageAdmissionReplay("operator", "delegated", "a" * 64)
    try:
        admitted = admission.admit(
            job_id="sj_0000000000000001",
            kind="analysis",
            actor="studio-service",
            workspace="default",
            request_id="trace-only",
            idempotency_key=None,
            experiment_sha256=None,
            admission={"budget": 3},
            execution_model="process",
            replay=replay,
            supervisor=api_generation,
        )
        row = (
            ledger.connection()
            .execute(
                "SELECT supervisor,state FROM admission_reservations WHERE job_id=?",
                (admitted.record.job_id,),
            )
            .fetchone()
        )
        assert row["supervisor"] == api_generation
        assert row["state"] == "running"
        assert admitted.record.lease_owner == api_generation
        assert admitted.record.lease_owner != ledger.supervisor

        admission.release(job_id=admitted.record.job_id)
        assert (
            ledger.connection()
            .execute(
                "SELECT COUNT(*) FROM admission_reservations WHERE job_id=?",
                (admitted.record.job_id,),
            )
            .fetchone()[0]
            == 1
        )
        admission.release(job_id=admitted.record.job_id, supervisor=api_generation)
        assert (
            ledger.connection()
            .execute(
                "SELECT COUNT(*) FROM admission_reservations WHERE job_id=?",
                (admitted.record.job_id,),
            )
            .fetchone()[0]
            == 0
        )
    finally:
        ledger.close()


def test_delegated_admission_refuses_unprovable_supervisor(tmp_path: Path) -> None:
    """A peer generation that cannot be established cannot own a job."""
    ledger = StudioJobLedger(root=tmp_path)
    admission = SharedJobAdmission(ledger, max_concurrent=1, max_queued=0)
    try:
        with pytest.raises(StudioJobRejected, match="not provably live"):
            admission.admit(
                job_id="sj_0000000000000001",
                kind="analysis",
                actor="studio-service",
                workspace="default",
                request_id=None,
                idempotency_key=None,
                experiment_sha256=None,
                admission=None,
                execution_model="process",
                supervisor="api:unverified",
            )
        assert ledger.list_records() == ()
        assert admission.snapshot().admitted == 0
    finally:
        ledger.close()


def test_delegated_replay_keeps_original_api_owner_across_storage_restart(
    tmp_path: Path,
) -> None:
    """A storage restart cannot turn its own generation into the prior lease."""
    replay = StorageAdmissionReplay("operator", "delegated-retry", "f" * 64)
    api_generation = supervisor_identity()
    first_ledger = StudioJobLedger(root=tmp_path, supervisor="storage:first")
    first_admission = SharedJobAdmission(first_ledger, max_concurrent=1, max_queued=0)
    original = _admit(
        first_admission,
        job_id="sj_0000000000000001",
        replay=replay,
        supervisor=api_generation,
    )
    first_ledger.close()

    second_ledger = StudioJobLedger(root=tmp_path, supervisor="storage:second")
    try:
        second_admission = SharedJobAdmission(second_ledger, max_concurrent=1, max_queued=0)
        repeated = _admit(
            second_admission,
            job_id="sj_0000000000000002",
            replay=replay,
            supervisor=api_generation,
        )
        assert repeated.duplicate
        assert repeated.record == original.record
        assert repeated.record.lease_owner == api_generation
        assert second_admission.snapshot().admitted == 1
    finally:
        second_ledger.close()


def _owner() -> subprocess.Popen[bytes]:
    """Start a real process standing in for the delegated API generation."""
    return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])


def _exit(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is None:
        process.kill()
    process.wait(timeout=10)


def test_queued_delegated_admission_releases_when_api_generation_exits(tmp_path: Path) -> None:
    """A queued API request cannot be admitted after its owner disappears.

    The owner process really exits when the transaction recording the queued
    request commits; the next queue pass observes it.
    """
    ledger = StudioJobLedger(root=tmp_path)
    admission = SharedJobAdmission(ledger, max_concurrent=1, max_queued=1)
    _admit(
        admission,
        job_id="sj_0000000000000001",
        replay=StorageAdmissionReplay("operator", "occupied", "a" * 64),
    )
    owner = _owner()
    statements: list[str] = []

    def exit_when_queued(statement: str) -> None:
        if statement == "COMMIT" and any("admission_reservations" in s for s in statements):
            _exit(owner)
        statements.append(statement)

    connection = ledger.connection()
    connection.set_trace_callback(exit_when_queued)
    try:
        with pytest.raises(StudioJobRejected, match="exited during admission"):
            _admit(
                admission,
                job_id="sj_0000000000000002",
                replay=StorageAdmissionReplay("operator", "queued", "b" * 64),
                supervisor=supervisor_identity(owner.pid),
            )
        assert owner.poll() is not None
        assert admission.snapshot().queued == 0
        assert admission.snapshot().admitted == 1
        assert len(ledger.list_records()) == 1
    finally:
        connection.set_trace_callback(None)
        _exit(owner)
        ledger.close()


def test_delegated_admission_rechecks_owner_after_opening_transaction(tmp_path: Path) -> None:
    """An API peer that exits while SQLite is reached cannot gain a slot.

    The owner process really exits as the admission transaction begins, after
    the checks made before it.
    """
    ledger = StudioJobLedger(root=tmp_path)
    admission = SharedJobAdmission(ledger, max_concurrent=1, max_queued=0)
    owner = _owner()

    def exit_on_begin(statement: str) -> None:
        if statement == "BEGIN IMMEDIATE":
            _exit(owner)

    connection = ledger.connection()
    connection.set_trace_callback(exit_on_begin)
    try:
        with pytest.raises(StudioJobRejected, match="exited during admission"):
            _admit(
                admission,
                job_id="sj_0000000000000001",
                replay=StorageAdmissionReplay("operator", "peer-exit", "a" * 64),
                supervisor=supervisor_identity(owner.pid),
            )
        assert ledger.list_records() == ()
        assert admission.snapshot().running == 0
        assert admission.snapshot().queued == 0
        assert admission.snapshot().admitted == 0
    finally:
        connection.set_trace_callback(None)
        _exit(owner)
        ledger.close()


def test_delegated_admission_refuses_peer_exit_while_sqlite_writer_waits(tmp_path: Path) -> None:
    """A real departed process cannot gain a job after a SQLite lock clears.

    The writer's own connection reports the start of its admission
    transaction, which follows the owner checks made before it; the owner then
    exits while the transaction waits for the held lock.
    """
    holder = StudioJobLedger(root=tmp_path)
    writer = StudioJobLedger(root=tmp_path)
    admission = SharedJobAdmission(writer, max_concurrent=1, max_queued=0)
    owner = _owner()
    generation = supervisor_identity(owner.pid)
    waiting = threading.Event()

    def report_begin(statement: str) -> None:
        if statement == "BEGIN IMMEDIATE":
            waiting.set()

    def submit() -> StudioJobSubmission:
        writer.connection().set_trace_callback(report_begin)
        try:
            return _admit(
                admission,
                job_id="sj_0000000000000001",
                replay=StorageAdmissionReplay("operator", "locked-peer-exit", "a" * 64),
                supervisor=generation,
            )
        finally:
            writer.close()

    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            with holder.transaction():
                future = pool.submit(submit)
                assert waiting.wait(timeout=10)
                _exit(owner)
            with pytest.raises(StudioJobRejected, match="exited during admission"):
                future.result(timeout=10)
        assert holder.list_records() == ()
        assert (
            holder.connection()
            .execute("SELECT COUNT(*) FROM storage_admission_replays")
            .fetchone()[0]
            == 0
        )
        assert admission.snapshot().admitted == 0
        assert admission.snapshot().queued == 0
    finally:
        _exit(owner)
        holder.close()
        writer.close()


def test_invalid_replay_identity_refuses_before_reservation(tmp_path: Path) -> None:
    """Malformed replay identity cannot create a capacity or job record."""
    ledger = StudioJobLedger(root=tmp_path)
    admission = SharedJobAdmission(ledger, max_concurrent=1, max_queued=0)
    cases = (
        StorageAdmissionReplay("", "id", "a" * 64),
        StorageAdmissionReplay("operator", " ", "a" * 64),
        StorageAdmissionReplay("operator", "line\nbreak", "a" * 64),
        StorageAdmissionReplay("operator", "id", "A" * 64),
        StorageAdmissionReplay("operator", "id", "a" * 63),
    )
    try:
        for replay in cases:
            with pytest.raises(ValueError, match="invalid storage admission replay identity"):
                _admit(admission, job_id="sj_0000000000000001", replay=replay)
        assert ledger.list_records() == ()
        assert admission.snapshot().admitted == 0
        assert (
            ledger.connection()
            .execute("SELECT COUNT(*) FROM storage_admission_replays")
            .fetchone()[0]
            == 0
        )
    finally:
        ledger.close()


def test_storage_replay_refuses_legacy_idempotency_lookup(tmp_path: Path) -> None:
    """A legacy key cannot return a job without checking its mutation digest."""
    ledger = StudioJobLedger(root=tmp_path)
    admission = SharedJobAdmission(ledger, max_concurrent=1, max_queued=0)
    try:
        with pytest.raises(ValueError, match="legacy idempotency key"):
            admission.admit(
                job_id="sj_0000000000000001",
                kind="analysis",
                actor="studio-service",
                workspace="default",
                request_id="trace-only",
                idempotency_key="browser-key",
                experiment_sha256=None,
                admission=None,
                execution_model="process",
                replay=StorageAdmissionReplay("operator", "browser-key", "a" * 64),
            )
        assert ledger.list_records() == ()
        assert admission.snapshot().admitted == 0
    finally:
        ledger.close()


def test_simultaneous_same_mutation_admits_only_one_job(tmp_path: Path) -> None:
    """Two real SQLite writers converge on one recorded admission outcome."""
    ledgers = (StudioJobLedger(root=tmp_path), StudioJobLedger(root=tmp_path))
    admissions = tuple(SharedJobAdmission(item, max_concurrent=1, max_queued=1) for item in ledgers)
    barrier = threading.Barrier(2)
    replay = StorageAdmissionReplay("operator", "simultaneous", "b" * 64)

    def submit(index: int) -> StudioJobSubmission:
        barrier.wait(timeout=3)
        return _admit(admissions[index], job_id=f"sj_{index + 1:016d}", replay=replay)

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(submit, index) for index in range(2)]
            outcomes = [future.result(timeout=5) for future in futures]
        assert sum(not outcome.duplicate for outcome in outcomes) == 1
        assert outcomes[0].record.job_id == outcomes[1].record.job_id
        assert len(ledgers[0].list_records()) == 1
        assert admissions[0].snapshot().admitted == 1
    finally:
        for ledger in ledgers:
            ledger.close()


def test_replay_refuses_snapshot_from_another_workspace(tmp_path: Path) -> None:
    """A corrupt replay row cannot return another workspace's job as success."""
    ledger = StudioJobLedger(root=tmp_path)
    admission = SharedJobAdmission(ledger, max_concurrent=1, max_queued=0)
    replay = StorageAdmissionReplay("operator", "workspace-bound", "b" * 64)
    try:
        original = _admit(admission, job_id="sj_0000000000000001", replay=replay)
        forged = replace(original.record, workspace="other").to_public_dict()
        with ledger.transaction() as connection:
            connection.execute(
                "UPDATE storage_admission_replays SET record_json=? WHERE workspace=? "
                "AND requester=? AND mutation_id=?",
                (json.dumps(forged), "default", replay.requester, replay.mutation_id),
            )
        with pytest.raises(StudioJobRejected, match="workspace is invalid"):
            _admit(admission, job_id="sj_0000000000000002", replay=replay)
        assert len(ledger.list_records()) == 1
        assert admission.snapshot().admitted == 1
    finally:
        ledger.close()


def test_refusal_replays_after_capacity_changes_without_second_counter(tmp_path: Path) -> None:
    """The original full-queue outcome persists even after a slot is released."""
    ledger = StudioJobLedger(root=tmp_path)
    admission = SharedJobAdmission(ledger, max_concurrent=1, max_queued=0)
    occupied = StorageAdmissionReplay("operator", "occupied", "c" * 64)
    refused = StorageAdmissionReplay("operator", "refused", "d" * 64)
    try:
        _admit(admission, job_id="sj_0000000000000001", replay=occupied)
        with pytest.raises(StudioJobQueueFull) as first:
            _admit(admission, job_id="sj_0000000000000002", replay=refused)
        assert admission.snapshot().refused == 1
        admission.release(job_id="sj_0000000000000001")
        with pytest.raises(StudioJobQueueFull) as second:
            _admit(admission, job_id="sj_0000000000000003", replay=refused)
        assert second.value.to_public_detail() == first.value.to_public_detail()
        assert admission.snapshot().refused == 1
        assert len(ledger.list_records()) == 1
    finally:
        ledger.close()


def test_queued_timeout_records_refusal_and_releases_its_reservation(tmp_path: Path) -> None:
    """A bounded waiting mutation leaves no occupied queue place on timeout."""
    ledger = StudioJobLedger(root=tmp_path)
    admission = SharedJobAdmission(ledger, max_concurrent=1, max_queued=1)
    replay = StorageAdmissionReplay("operator", "queued-timeout", "e" * 64)
    try:
        _admit(
            admission,
            job_id="sj_0000000000000001",
            replay=StorageAdmissionReplay("operator", "occupied", "d" * 64),
        )
        with pytest.raises(StudioJobQueueFull) as first:
            admission.admit(
                job_id="sj_0000000000000002",
                kind="analysis",
                actor="studio-service",
                workspace="default",
                request_id="trace-only",
                idempotency_key=None,
                experiment_sha256=None,
                admission=None,
                execution_model="process",
                timeout_seconds=0.15,
                replay=replay,
            )
        assert admission.snapshot().queued == 0
        assert admission.snapshot().refused == 1
        with pytest.raises(StudioJobQueueFull) as second:
            _admit(admission, job_id="sj_0000000000000003", replay=replay)
        assert second.value.to_public_detail() == first.value.to_public_detail()
        assert admission.snapshot().refused == 1
    finally:
        ledger.close()


def test_replay_write_failure_rolls_back_job_capacity_and_transition(tmp_path: Path) -> None:
    """The replay outcome and accepted job never commit separately."""
    ledger = StudioJobLedger(root=tmp_path)
    admission = SharedJobAdmission(ledger, max_concurrent=1, max_queued=0)
    replay = StorageAdmissionReplay("operator", "atomic", "f" * 64)
    with ledger.transaction() as connection:
        connection.execute(
            "CREATE TRIGGER refuse_replay BEFORE INSERT ON storage_admission_replays "
            "BEGIN SELECT RAISE(ABORT,'injected replay refusal'); END"
        )
    try:
        with pytest.raises(sqlite3.IntegrityError, match="injected replay refusal"):
            _admit(admission, job_id="sj_0000000000000001", replay=replay)
        assert ledger.list_records() == ()
        assert (
            ledger.connection().execute("SELECT COUNT(*) FROM job_transitions").fetchone()[0] == 0
        )
        assert admission.snapshot().admitted == 0
        assert admission.snapshot().running == 0
        with ledger.transaction() as connection:
            connection.execute("DROP TRIGGER refuse_replay")
        retry = _admit(admission, job_id="sj_0000000000000002", replay=replay)
        assert retry.record.job_id == "sj_0000000000000002"
        assert admission.snapshot().admitted == 1
    finally:
        ledger.close()


def test_v5_migration_adds_replay_without_relabeling_existing_jobs(tmp_path: Path) -> None:
    """An existing ledger gains an empty replay table and retains its rows."""
    ledger = StudioJobLedger(root=tmp_path)
    first = SharedJobAdmission(ledger, max_concurrent=1, max_queued=0)
    original = _admit(
        first,
        job_id="sj_0000000000000001",
        replay=StorageAdmissionReplay("operator", "before", "e" * 64),
    ).record
    ledger.close()
    path = tmp_path / LEDGER_FILENAME
    with sqlite3.connect(path) as connection:
        connection.execute("DROP TABLE storage_admission_replays")
        connection.execute("UPDATE schema_meta SET value='5' WHERE key='schema_version'")
        connection.execute(
            "UPDATE schema_meta SET value='studio.job-ledger.v5' WHERE key='schema_name'"
        )
    migrated = StudioJobLedger(root=tmp_path)
    try:
        assert migrated.record(original.job_id) == original
        assert (
            migrated.connection()
            .execute("SELECT COUNT(*) FROM storage_admission_replays")
            .fetchone()[0]
            == 0
        )
        assert migrated.connection().execute(
            "SELECT value FROM schema_meta WHERE key='schema_version'"
        ).fetchone()[0] == str(SCHEMA_VERSION)
    finally:
        migrated.close()
