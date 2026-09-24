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
from contextlib import closing
from datetime import timedelta
from pathlib import Path

import pytest

from sc_neurocore.studio.platform.jobs_ledger_schema import (
    StudioJobLedgerCorrupt,
)
from sc_neurocore.studio.platform.jobs_models import (
    StudioJobArtifact,
    StudioJobRejected,
    StudioJobStatus,
)

from tests.test_studio_jobs_ledger import _ledger, _admit, UTC_CLOCK_START


class TestStateMachine:
    def test_recovery_does_not_hide_corrupt_artifacts_as_a_purge(self, tmp_path: Path) -> None:
        """Only an absent job is omitted; malformed retained evidence remains an error.

        Right after recovery read its snapshot of live rows, a second real
        connection corrupts the job's stored artifacts, as damage between reads
        would.
        """
        ledger = _ledger(tmp_path)
        submitted = _admit(ledger)
        statements: list[str] = []

        def corrupt_after_snapshot(statement: str) -> None:
            if statements and statements[-1].startswith("SELECT * FROM jobs WHERE status IN"):
                with closing(sqlite3.connect(ledger.path, isolation_level=None)) as other:
                    other.execute(
                        "UPDATE jobs SET artifacts = ? WHERE job_id = ?",
                        ("[{}]", submitted.record.job_id),
                    )
            statements.append(statement)

        connection = ledger.connection()
        connection.set_trace_callback(corrupt_after_snapshot)
        try:
            with pytest.raises(StudioJobLedgerCorrupt, match="relative_path"):
                ledger.reconcile()
        finally:
            connection.set_trace_callback(None)

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


@pytest.mark.parametrize("stored", ["null", "false", "0", '""', "{}", '"manifest"', "[", "[null]"])
def test_corrupt_artifact_column_cannot_masquerade_as_an_empty_manifest(
    tmp_path: Path, stored: str
) -> None:
    """Public reads and reconciliation refuse corrupt custody without rewriting its evidence."""
    ledger = _ledger(tmp_path)
    try:
        submission = _admit(ledger)
        job_id = submission.record.job_id
        assert ledger.record(job_id).artifacts == ()
        history = ledger.transitions(job_id)
        with ledger.transaction() as connection:
            connection.execute("UPDATE jobs SET artifacts=? WHERE job_id=?", (stored, job_id))
        before = (
            ledger.connection().execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        )
        with pytest.raises(StudioJobLedgerCorrupt):
            ledger.record(job_id)
        with pytest.raises(StudioJobLedgerCorrupt):
            ledger.list_records()
        with pytest.raises(StudioJobLedgerCorrupt):
            ledger.reconcile()
        assert ledger.transitions(job_id) == history
        after = (
            ledger.connection().execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        )
        assert after == before
    finally:
        ledger.close()
