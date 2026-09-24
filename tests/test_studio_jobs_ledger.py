# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio job ledger contract

"""The durable store: what it accepts, what it refuses, what it remembers."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

import pytest

from sc_neurocore.studio.platform.jobs_ledger import StudioJobLedger
from sc_neurocore.studio.platform.jobs_ledger_schema import (
    JOB_LEDGER_SCHEMA_VERSION,
    LEDGER_FILENAME,
    SCHEMA_VERSION,
    SCHEMA_V1,
    StudioJobLedgerCorrupt,
    StudioJobSubmission,
)
from sc_neurocore.studio.training_contract import resolve_training_config
from sc_neurocore.studio.platform.jobs import StudioJobManager
from sc_neurocore.studio.platform.jobs_models import (
    StudioJobRejected,
)

UTC_CLOCK_START = datetime.fromisoformat("2026-09-06T00:00:00+00:00")


def _ledger(root: Path, **kwargs: object) -> StudioJobLedger:
    return StudioJobLedger(root=root, **kwargs)  # type: ignore[arg-type]


def _job_manager(root: Path) -> StudioJobManager:
    """Open a manager over the shared root, reconciling on construction."""
    return StudioJobManager(
        root=root, allowed_kinds=frozenset({"analysis"}), default_timeout_seconds=30.0
    )


def _admit(
    ledger: StudioJobLedger, job_id: str = "sj_0000000000000001", **kwargs: object
) -> StudioJobSubmission:
    fields: dict[str, object] = {
        "job_id": job_id,
        "kind": "analysis",
        "actor": "alice",
        "workspace": "default",
        "request_id": None,
        "idempotency_key": None,
        "experiment_sha256": None,
        "admission": None,
        "execution_model": "thread",
    }
    fields.update(kwargs)
    return ledger.create(**fields)  # type: ignore[arg-type]


class TestSchema:
    def test_a_new_ledger_records_its_schema_version(self, tmp_path: Path) -> None:
        ledger = _ledger(tmp_path)
        rows = dict(ledger.connection().execute("SELECT key, value FROM schema_meta").fetchall())
        assert rows["schema_version"] == str(SCHEMA_VERSION)
        assert rows["schema_name"] == JOB_LEDGER_SCHEMA_VERSION
        assert ledger.path.name == "job_ledger.sqlite3"

    def test_reopening_a_ledger_keeps_its_jobs(self, tmp_path: Path) -> None:
        first = _ledger(tmp_path)
        _admit(first)
        first.close()

        assert len(_ledger(tmp_path).list_records()) == 1

    def test_v6_ledger_gains_nullable_training_snapshot(self, tmp_path: Path) -> None:
        """A real prior schema opens without inventing historical configuration."""
        old_schema = SCHEMA_V1.replace("    training_config TEXT,\n", "")
        assert old_schema != SCHEMA_V1
        with sqlite3.connect(tmp_path / LEDGER_FILENAME) as connection:
            connection.executescript(old_schema)
            connection.execute("INSERT INTO schema_meta VALUES ('schema_version','6')")
            connection.execute(
                "INSERT INTO schema_meta VALUES ('schema_name','studio.job-ledger.v6')"
            )
        ledger = _ledger(tmp_path)
        try:
            legacy = _admit(ledger, kind="training", execution_model="process").record
            assert legacy.training_config is None
            assert ledger.connection().execute(
                "SELECT value FROM schema_meta WHERE key='schema_version'"
            ).fetchone()[0] == str(SCHEMA_VERSION)
        finally:
            ledger.close()

    def test_a_ledger_from_the_future_is_refused_rather_than_downgraded(
        self, tmp_path: Path
    ) -> None:
        ledger = _ledger(tmp_path)
        with ledger.transaction() as connection:
            connection.execute(
                "UPDATE schema_meta SET value = ? WHERE key = 'schema_version'",
                (str(SCHEMA_VERSION + 1),),
            )
        ledger.close()

        with pytest.raises(StudioJobLedgerCorrupt) as refusal:
            _ledger(tmp_path)
        assert "Upgrade the package" in str(refusal.value)

    def test_transitions_cannot_be_edited_or_erased(self, tmp_path: Path) -> None:
        ledger = _ledger(tmp_path)
        _admit(ledger)
        connection = ledger.connection()

        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute("UPDATE job_transitions SET to_status = 'completed'")
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute("DELETE FROM job_transitions")


class TestAdmission:
    def test_training_snapshot_is_canonical_bounded_and_durable(self, tmp_path: Path) -> None:
        """Admission binds a resolved config and refuses oversized or wrong-kind values."""
        ledger = _ledger(tmp_path)
        config = resolve_training_config(
            {"epochs": 1, "batch_size": 64, "hidden": [4], "timesteps": 1}
        ).to_public_dict()
        first = _admit(
            ledger, kind="training", execution_model="process", training_config=config
        ).record
        assert first.training_config == config
        ledger.close()
        assert _ledger(tmp_path).record(first.job_id).training_config == config

        with pytest.raises(ValueError, match="Only a training job"):
            _admit(ledger, job_id="sj_0000000000000002", training_config=config)
        with pytest.raises(ValueError, match="4096-byte"):
            _admit(
                ledger,
                job_id="sj_0000000000000003",
                kind="training",
                training_config={**config, "hidden": [1] * 2000},
            )
        assert [record.job_id for record in ledger.list_records()] == [first.job_id]
        with ledger.transaction() as connection:
            connection.execute(
                "UPDATE jobs SET training_config=? WHERE job_id=?",
                ('{"epochs":0}', first.job_id),
            )
        with pytest.raises(StudioJobLedgerCorrupt, match="training configuration"):
            ledger.record(first.job_id)

    def test_admission_records_the_custody_fields(self, tmp_path: Path) -> None:
        ledger = _ledger(tmp_path)

        submission = _admit(
            ledger,
            workspace="lab-a",
            request_id="req-7",
            idempotency_key="key-7",
            experiment_sha256="a" * 64,
            admission={"route": "POST /api/analysis/jobs", "budget": "synchronous"},
        )

        assert submission.duplicate is False
        record = submission.record
        assert record.status == "pending"
        assert record.workspace == "lab-a"
        assert record.idempotency_key == "key-7"
        assert record.experiment_sha256 == "a" * 64
        assert record.admission["route"] == "POST /api/analysis/jobs"
        assert record.lease_owner == ledger.supervisor
        assert record.heartbeat_at_utc is not None

    def test_the_same_key_is_admitted_once(self, tmp_path: Path) -> None:
        ledger = _ledger(tmp_path)
        first = _admit(ledger, job_id="sj_0000000000000001", idempotency_key="key-1")

        second = _admit(ledger, job_id="sj_0000000000000002", idempotency_key="key-1")

        assert second.duplicate is True
        assert second.record.job_id == first.record.job_id
        assert len(ledger.list_records()) == 1

    def test_the_same_key_in_another_workspace_is_a_different_job(self, tmp_path: Path) -> None:
        ledger = _ledger(tmp_path)
        _admit(ledger, job_id="sj_0000000000000001", idempotency_key="key-1", workspace="a")

        other = _admit(ledger, job_id="sj_0000000000000002", idempotency_key="key-1", workspace="b")

        assert other.duplicate is False
        assert len(ledger.list_records()) == 2

    def test_a_job_without_a_key_is_never_deduplicated(self, tmp_path: Path) -> None:
        ledger = _ledger(tmp_path)
        _admit(ledger, job_id="sj_0000000000000001")
        _admit(ledger, job_id="sj_0000000000000002")
        assert len(ledger.list_records()) == 2


class TestIsolation:
    def test_a_read_scoped_to_another_actor_finds_nothing(self, tmp_path: Path) -> None:
        ledger = _ledger(tmp_path)
        _admit(ledger, actor="alice")

        assert ledger.record("sj_0000000000000001", actor="alice").owner == "alice"
        with pytest.raises(KeyError):
            ledger.record("sj_0000000000000001", actor="bob")
        assert ledger.list_records(actor="bob") == ()

    def test_a_read_scoped_to_another_workspace_finds_nothing(self, tmp_path: Path) -> None:
        ledger = _ledger(tmp_path)
        _admit(ledger, workspace="lab-a")

        with pytest.raises(KeyError):
            ledger.record("sj_0000000000000001", workspace="lab-b")
        assert ledger.list_records(workspace="lab-a") != ()


class TestPurge:
    def test_only_a_terminal_job_can_be_purged(self, tmp_path: Path) -> None:
        ledger = _ledger(tmp_path)
        _admit(ledger)
        ledger.transition("sj_0000000000000001", "running")

        with pytest.raises(StudioJobRejected, match="not terminal"):
            ledger.delete("sj_0000000000000001")

        ledger.transition("sj_0000000000000001", "completed")
        ledger.delete("sj_0000000000000001")
        assert ledger.list_records() == ()
        assert ledger.transitions("sj_0000000000000001") == ()
        with pytest.raises(KeyError):
            ledger.delete("sj_0000000000000001")
        assert ledger.list_records() == ()
