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
from typing import Any

import pytest

from sc_neurocore.studio.platform.jobs_ledger import StudioJobLedger
from sc_neurocore.studio.platform.jobs_ledger_schema import (
    JOB_LEDGER_SCHEMA_VERSION,
    SCHEMA_VERSION,
    StudioJobLedgerCorrupt,
    StudioJobSubmission,
)
from sc_neurocore.studio.platform.jobs import StudioJobManager
from sc_neurocore.studio.platform.jobs_context import StudioJobContext
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


class TestArtifactCustody:
    @pytest.mark.parametrize(
        "replacement",
        [
            {"result": {"written": False}},
            {"artifacts": ()},
            {"error": "late error"},
            {"started_at_utc": "2026-01-01T00:00:00Z"},
            {"finished_at_utc": "2026-01-01T00:00:00Z"},
        ],
        ids=["result", "manifest", "error", "start", "finish"],
    )
    def test_late_writer_cannot_change_a_real_completed_job(
        self, tmp_path: Path, replacement: dict[str, Any]
    ) -> None:
        """Independent ledger writers cannot revise the real runner's sealed evidence."""
        root = tmp_path / "jobs"
        manager = _job_manager(root)

        def task(context: StudioJobContext) -> dict[str, object]:
            context.write_artifact("result.bin", b"original payload")
            return {"written": True}

        submitted = manager.submit(kind="analysis", owner="alice", request_id=None, task=task)
        sealed = manager.wait(submitted.job_id, timeout_seconds=5.0)
        assert sealed.status == "completed"
        writer = _ledger(root)
        history = writer.transitions(sealed.job_id)
        with pytest.raises(StudioJobRejected, match="cannot rewrite"):
            writer.transition(sealed.job_id, "completed", **replacement)
        assert manager.record(sealed.job_id) == sealed
        assert writer.transitions(sealed.job_id) == history
        assert manager.read_artifact(sealed.job_id, "result.bin").payload == b"original payload"
        assert (
            writer.transition(
                sealed.job_id,
                "completed",
                started_at_utc=sealed.started_at_utc,
                finished_at_utc=sealed.finished_at_utc,
                error=sealed.error,
                result=sealed.result,
                artifacts=sealed.artifacts,
            )
            == sealed
        )

    def test_a_completed_job_keeps_its_manifest_across_a_restart(self, tmp_path: Path) -> None:
        root = tmp_path / "jobs"
        manager = _job_manager(root)

        def task(context: StudioJobContext) -> dict[str, object]:
            context.write_artifact("result.bin", b"payload")
            return {"written": True}

        record = manager.submit(kind="analysis", owner="alice", request_id="req-3", task=task)
        done = manager.wait(record.job_id, 30.0)
        assert done.status == "completed"
        assert done.artifacts != ()

        restarted = _job_manager(root)
        recovered = restarted.record(record.job_id)

        assert recovered.artifacts == done.artifacts
        assert restarted.read_artifact(record.job_id, "result.bin").payload == b"payload"
