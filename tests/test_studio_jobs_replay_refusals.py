# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Durable admission replay refusals

"""Refuse damaged retained replies without admitting another job."""

import sqlite3
from pathlib import Path

import pytest

from sc_neurocore.studio.platform.jobs_admission_replay import StorageAdmissionReplay
from sc_neurocore.studio.platform.jobs_admission import StudioJobQueueFull
from sc_neurocore.studio.platform.jobs_ledger import StudioJobLedger
from sc_neurocore.studio.platform.jobs_ledger_schema import StudioJobSubmission
from sc_neurocore.studio.platform.jobs_models import StudioJobRefused
from sc_neurocore.studio.platform.jobs_shared_admission import SharedJobAdmission
from sc_neurocore.refusals import AuthoredRefusal


def _admit(
    admission: SharedJobAdmission, *, job_id: str, replay: StorageAdmissionReplay
) -> StudioJobSubmission:
    return admission.admit(
        job_id=job_id,
        kind="analysis",
        actor="operator",
        workspace="default",
        request_id=None,
        idempotency_key=None,
        experiment_sha256=None,
        admission=None,
        execution_model="process",
        replay=replay,
    )


@pytest.mark.parametrize("damaged", ["not JSON", "null", "[]"])
def test_damaged_retained_reply_is_an_authored_refusal(tmp_path: Path, damaged: str) -> None:
    """A real SQLite replay fault preserves its original job and occupied slot."""
    ledger = StudioJobLedger(root=tmp_path / "jobs")
    admission = SharedJobAdmission(ledger, max_concurrent=1, max_queued=0)
    replay = StorageAdmissionReplay("operator", "retained-reply", "a" * 64)
    try:
        original = _admit(admission, job_id="sj_0000000000000001", replay=replay)
        with ledger.transaction() as connection:
            connection.execute(
                "UPDATE storage_admission_replays SET record_json=?",
                (damaged,),
            )
        before = admission.snapshot()
        transitions = ledger.transitions(original.record.job_id)
        with pytest.raises(StudioJobRefused) as refused:
            _admit(admission, job_id="sj_0000000000000002", replay=replay)
        assert str(refused.value) == "Stored admission replay is invalid."
        assert ledger.list_records() == (original.record,)
        assert ledger.transitions(original.record.job_id) == transitions
        assert admission.snapshot() == before
        retained = (
            ledger.connection()
            .execute("SELECT record_json FROM storage_admission_replays")
            .fetchone()
        )
        assert retained[0] == damaged
    finally:
        ledger.close()


def test_conflicting_root_limits_do_not_create_another_reservation(tmp_path: Path) -> None:
    """Another admission controller cannot silently change an occupied root's limits."""
    ledger = StudioJobLedger(root=tmp_path / "jobs")
    first = SharedJobAdmission(ledger, max_concurrent=1, max_queued=0)
    second = SharedJobAdmission(ledger, max_concurrent=2, max_queued=0)
    try:
        original = _admit(
            first,
            job_id="sj_0000000000000001",
            replay=StorageAdmissionReplay("operator", "original", "a" * 64),
        )
        before = first.snapshot()
        with pytest.raises(StudioJobRefused) as refused:
            _admit(
                second,
                job_id="sj_0000000000000002",
                replay=StorageAdmissionReplay("operator", "conflicting", "b" * 64),
            )
        assert str(refused.value) == "Studio job root has different admission limits."
        assert ledger.list_records() == (original.record,)
        assert first.snapshot() == before
    finally:
        ledger.close()


def test_reusing_an_occupied_identifier_preserves_the_first_reservation(tmp_path: Path) -> None:
    """A different mutation cannot replace a live job with the same identifier."""
    ledger = StudioJobLedger(root=tmp_path / "jobs")
    admission = SharedJobAdmission(ledger, max_concurrent=1, max_queued=0)
    try:
        original = _admit(
            admission,
            job_id="sj_0000000000000001",
            replay=StorageAdmissionReplay("operator", "first", "a" * 64),
        )
        before = admission.snapshot()
        with pytest.raises(StudioJobRefused) as refused:
            _admit(
                admission,
                job_id=original.record.job_id,
                replay=StorageAdmissionReplay("operator", "second", "b" * 64),
            )
        assert str(refused.value) == "Studio reservation identifier is already occupied."
        assert ledger.list_records() == (original.record,)
        assert admission.snapshot() == before
        assert (
            ledger.connection()
            .execute("SELECT COUNT(*) FROM storage_admission_replays")
            .fetchone()[0]
            == 1
        )
    finally:
        ledger.close()


def test_incomplete_retained_capacity_refusal_cannot_be_replayed(tmp_path: Path) -> None:
    """SQLite refuses missing counters; replay retains the exact original queue refusal."""
    ledger = StudioJobLedger(root=tmp_path / "jobs")
    admission = SharedJobAdmission(ledger, max_concurrent=1, max_queued=0)
    replay = StorageAdmissionReplay("operator", "overflow", "b" * 64)
    try:
        original = _admit(
            admission,
            job_id="sj_0000000000000001",
            replay=StorageAdmissionReplay("operator", "running", "a" * 64),
        )
        with pytest.raises(StudioJobQueueFull) as initial:
            _admit(admission, job_id="sj_0000000000000002", replay=replay)
        with (
            pytest.raises(sqlite3.IntegrityError, match="CHECK constraint failed"),
            ledger.transaction() as connection,
        ):
            connection.execute(
                "UPDATE storage_admission_replays SET running=NULL WHERE mutation_id=?",
                (replay.mutation_id,),
            )
        before = admission.snapshot()
        with pytest.raises(StudioJobQueueFull) as refused:
            _admit(admission, job_id="sj_0000000000000003", replay=replay)
        assert (
            (refused.value.running, refused.value.queued, refused.value.limit)
            == (
                initial.value.running,
                initial.value.queued,
                initial.value.limit,
            )
            == (1, 0, 0)
        )
        assert ledger.list_records() == (original.record,)
        assert admission.snapshot() == before
        row = (
            ledger.connection()
            .execute(
                "SELECT outcome,running FROM storage_admission_replays WHERE mutation_id=?",
                (replay.mutation_id,),
            )
            .fetchone()
        )
        assert tuple(row) == ("refused", 1)
    finally:
        ledger.close()


def test_a_real_disk_bit_flip_refuses_replay_without_new_admission(tmp_path: Path) -> None:
    """A damaged retained outcome fails closed while SQLite constraints remain enabled."""
    root = tmp_path / "jobs"
    ledger = StudioJobLedger(root=root)
    admission = SharedJobAdmission(ledger, max_concurrent=1, max_queued=0)
    replay = StorageAdmissionReplay("operator", "disk-fault", "b" * 64)
    try:
        original = _admit(
            admission,
            job_id="sj_0000000000000001",
            replay=StorageAdmissionReplay("operator", "running", "a" * 64),
        )
        with pytest.raises(StudioJobQueueFull):
            _admit(admission, job_id="sj_0000000000000002", replay=replay)
        before = admission.snapshot()
        history = ledger.transitions(original.record.job_id)
        connection = ledger.connection()
        page_number, schema = connection.execute(
            "SELECT rootpage,sql FROM sqlite_schema WHERE name='storage_admission_replays'"
        ).fetchone()
        page_size = int(connection.execute("PRAGMA page_size").fetchone()[0])
        assert connection.execute("PRAGMA ignore_check_constraints").fetchone()[0] == 0
        assert connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()[0] == 0
        path = ledger.path
    finally:
        ledger.close()

    # SQLite stores text column bytes consecutively in each table-leaf record body:
    # https://www.sqlite.org/fileformat.html#record_format
    image = path.read_bytes()
    page_start = (int(page_number) - 1) * page_size
    page = image[page_start : page_start + page_size]
    assert page[0] == 0x0D
    needle = (
        replay.requester.encode()
        + replay.mutation_id.encode()
        + replay.payload_sha256.encode()
        + b"refused"
    )
    assert page.count(needle) == 1
    offset = page_start + page.index(needle) + len(needle) - len(b"refused")
    assert image[offset] == ord("r")
    with path.open("r+b") as file:
        file.seek(offset)
        assert file.write(bytes([image[offset] ^ 1])) == 1
    damaged = path.read_bytes()
    assert damaged[:offset] == image[:offset] and damaged[offset + 1 :] == image[offset + 1 :]
    assert damaged[offset] ^ image[offset] == 1

    reopened = StudioJobLedger(root=root)
    try:
        restored = SharedJobAdmission(reopened, max_concurrent=1, max_queued=0)
        connection = reopened.connection()
        assert connection.execute("PRAGMA ignore_check_constraints").fetchone()[0] == 0
        assert (
            connection.execute(
                "SELECT sql FROM sqlite_schema WHERE name='storage_admission_replays'"
            ).fetchone()[0]
            == schema
        )
        with pytest.raises(StudioJobRefused) as refused:
            _admit(restored, job_id="sj_0000000000000003", replay=replay)
        assert isinstance(refused.value, AuthoredRefusal)
        assert str(refused.value) == "Stored admission replay is invalid."
        assert reopened.list_records() == (original.record,)
        assert reopened.transitions(original.record.job_id) == history
        assert restored.snapshot() == before
        assert (
            connection.execute(
                "SELECT outcome FROM storage_admission_replays WHERE mutation_id=?",
                (replay.mutation_id,),
            ).fetchone()[0]
            == "sefused"
        )
    finally:
        reopened.close()
