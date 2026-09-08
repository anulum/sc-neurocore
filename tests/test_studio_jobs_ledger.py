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
from datetime import datetime, timedelta
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
    StudioJobArtifact,
    StudioJobRejected,
    StudioJobStatus,
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


class TestStateMachine:
    def test_recovery_propagates_unrelated_lookup_errors(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failed lookup other than the job itself must not masquerade as a purge."""
        ledger = _ledger(tmp_path)
        submitted = _admit(ledger)
        before = ledger.record(submitted.record.job_id)
        history = ledger.transitions(before.job_id)

        def broken_read(job_id: str) -> None:
            raise KeyError("unrelated_record_field")

        with monkeypatch.context() as patch:
            patch.setattr(ledger, "record", broken_read)
            with pytest.raises(KeyError, match="unrelated_record_field"):
                ledger.reconcile()
        assert ledger.record(before.job_id) == before
        assert ledger.transitions(before.job_id) == history

    def test_recovery_does_not_hide_corrupt_artifacts_as_a_purge(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Only an absent job is omitted; malformed retained evidence remains an error."""
        ledger = _ledger(tmp_path)
        submitted = _admit(ledger)
        live_rows = ledger.live_rows

        def snapshot_then_corrupt() -> tuple[sqlite3.Row, ...]:
            rows = live_rows()
            with ledger.transaction() as connection:
                connection.execute(
                    "UPDATE jobs SET artifacts = ? WHERE job_id = ?",
                    ("[{}]", submitted.record.job_id),
                )
            return rows

        monkeypatch.setattr(ledger, "live_rows", snapshot_then_corrupt)
        with pytest.raises(StudioJobLedgerCorrupt, match="relative_path"):
            ledger.reconcile()

    @pytest.mark.parametrize("change", ["none", "heartbeat", "result-type", "completed"])
    def test_conditional_transition_checks_the_observed_record(
        self, tmp_path: Path, change: str
    ) -> None:
        """A stale recovery decision cannot replace newer state, lease or typed data."""
        moment = [UTC_CLOCK_START]
        ledger = _ledger(tmp_path, clock=lambda: moment[0])
        submitted = _admit(ledger)
        expected = ledger.transition(submitted.record.job_id, "running", result={"value": 1})
        if change == "heartbeat":
            moment[0] += timedelta(seconds=10)
            ledger.heartbeat(expected.job_id)
        elif change == "result-type":
            ledger.transition(expected.job_id, "running", result={"value": True})
        elif change == "completed":
            ledger.transition(expected.job_id, "completed", result={"value": 42})
        current = ledger.record(expected.job_id)
        history = ledger.transitions(expected.job_id)
        observed = ledger.transition(
            expected.job_id, "interrupted", expected_record=expected, reason="recovery"
        )
        if change == "none":
            assert observed.status == "interrupted"
            assert len(ledger.transitions(expected.job_id)) == len(history) + 1
        else:
            assert observed == current
            assert ledger.transitions(expected.job_id) == history
        assert _ledger(tmp_path).record(expected.job_id) == observed

    @pytest.mark.parametrize(
        "status", ["completed", "failed", "cancelled", "timed_out", "interrupted"]
    )
    def test_repeating_a_terminal_status_preserves_the_entire_record(
        self, tmp_path: Path, status: StudioJobStatus
    ) -> None:
        """An idempotent terminal retry changes neither fields nor audit history."""
        moment = [UTC_CLOCK_START]
        ledger = _ledger(tmp_path, clock=lambda: moment[0])
        submitted = _admit(ledger)
        ledger.transition(submitted.record.job_id, "running")
        sealed = ledger.transition(submitted.record.job_id, status, result={"answer": 42})
        history = ledger.transitions(sealed.job_id)
        moment[0] += timedelta(seconds=20)
        assert ledger.transition(sealed.job_id, status) == sealed
        assert ledger.transition(sealed.job_id, status, result={"answer": 42}) == sealed
        assert _ledger(tmp_path).record(sealed.job_id) == sealed
        assert ledger.transitions(sealed.job_id) == history

    @pytest.mark.parametrize(
        "status", ["completed", "failed", "cancelled", "timed_out", "interrupted"]
    )
    def test_same_status_cannot_replace_a_terminal_result(
        self, tmp_path: Path, status: StudioJobStatus
    ) -> None:
        """A late writer cannot replace sealed evidence without a new transition."""
        ledger = _ledger(tmp_path)
        submitted = _admit(ledger)
        ledger.transition(submitted.record.job_id, "running")
        sealed = ledger.transition(submitted.record.job_id, status, result={"answer": 42})
        history = ledger.transitions(sealed.job_id)
        with pytest.raises(StudioJobRejected, match="terminal"):
            ledger.transition(sealed.job_id, status, result={"answer": 0})
        assert _ledger(tmp_path).record(sealed.job_id) == sealed
        assert ledger.transitions(sealed.job_id) == history

    def test_a_transition_is_appended_with_its_reason(self, tmp_path: Path) -> None:
        ledger = _ledger(tmp_path)
        _admit(ledger)

        ledger.transition("sj_0000000000000001", "running", reason="supervised")
        ledger.transition(
            "sj_0000000000000001",
            "completed",
            reason="supervised",
            result={"answer": 42},
            artifacts=(StudioJobArtifact(relative_path="out.bin", size_bytes=2, sha256="ab"),),
        )

        history = ledger.transitions("sj_0000000000000001")
        assert [entry["to_status"] for entry in history] == ["pending", "running", "completed"]
        assert [entry["sequence"] for entry in history] == [0, 1, 2]
        record = ledger.record("sj_0000000000000001")
        assert record.result == {"answer": 42}
        assert record.artifacts[0].relative_path == "out.bin"
        # A terminal job holds no lease.
        assert record.lease_owner is None
        assert record.lease_expires_at_utc is None

    @pytest.mark.parametrize(
        ("route", "attempt"),
        [
            (("running", "completed"), "running"),
            (("running", "failed"), "completed"),
            (("unknown", "interrupted"), "completed"),
            (("cancelling", "cancelled"), "running"),
        ],
        ids=["completed", "failed", "interrupted", "cancelled"],
    )
    def test_a_terminal_record_is_never_rewritten(
        self, tmp_path: Path, route: tuple[str, ...], attempt: str
    ) -> None:
        ledger = _ledger(tmp_path)
        _admit(ledger)
        for status in route:
            ledger.transition("sj_0000000000000001", status)  # type: ignore[arg-type]

        with pytest.raises(StudioJobRejected, match="cannot move"):
            ledger.transition("sj_0000000000000001", attempt)  # type: ignore[arg-type]

    def test_a_job_cannot_complete_without_running(self, tmp_path: Path) -> None:
        ledger = _ledger(tmp_path)
        _admit(ledger)
        with pytest.raises(StudioJobRejected, match="from 'pending' to 'completed'"):
            ledger.transition("sj_0000000000000001", "completed")

    def test_a_supervisor_starting_a_cancelling_job_keeps_the_cancellation(
        self, tmp_path: Path
    ) -> None:
        ledger = _ledger(tmp_path)
        _admit(ledger)
        ledger.transition("sj_0000000000000001", "cancelling", reason="requested")

        record = ledger.transition(
            "sj_0000000000000001", "running", started_at_utc="2026-09-06T00:00:01Z"
        )

        assert record.status == "cancelling"
        assert record.started_at_utc == "2026-09-06T00:00:01Z"
        assert [entry["to_status"] for entry in ledger.transitions("sj_0000000000000001")] == [
            "pending",
            "cancelling",
        ]

    def test_an_unknown_job_is_absent_rather_than_invented(self, tmp_path: Path) -> None:
        ledger = _ledger(tmp_path)
        with pytest.raises(KeyError):
            ledger.transition("sj_0000000000000009", "running")
        with pytest.raises(KeyError):
            ledger.record("sj_0000000000000009")


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
