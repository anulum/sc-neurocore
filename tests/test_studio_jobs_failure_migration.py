# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Legacy job diagnostic migration

"""Upgrade actual SQLite ledgers without rewriting diagnostics or transition history."""

import sqlite3
from pathlib import Path

from sc_neurocore.studio.platform.jobs_failures import GENERIC_JOB_FAILURE, authored_job_error
from sc_neurocore.studio.platform.jobs_ledger import StudioJobLedger


def test_v8_diagnostic_is_retained_without_public_promotion(tmp_path: Path) -> None:
    """A v8-shaped real database upgrades atomically and exposes a fixed fallback."""
    ledger = StudioJobLedger(root=tmp_path)
    job_id = "sj_0123456789abcdef"
    ledger.create(
        job_id=job_id,
        kind="analysis",
        actor="operator",
        workspace="default",
        request_id=None,
        idempotency_key=None,
        experiment_sha256=None,
        admission=None,
        execution_model="thread",
    )
    diagnostic = "filesystem failed at " + str(tmp_path / "private.bin")
    ledger.transition(job_id, "failed", error=diagnostic)
    transitions = ledger.transitions(job_id)
    ledger.close()
    with sqlite3.connect(tmp_path / "job_ledger.sqlite3") as connection:
        connection.execute("ALTER TABLE jobs DROP COLUMN public_error")
        connection.execute("UPDATE schema_meta SET value='8' WHERE key='schema_version'")
        connection.execute(
            "UPDATE schema_meta SET value='studio.job-ledger.v8' WHERE key='schema_name'"
        )
    restored = StudioJobLedger(root=tmp_path)
    record = restored.record(job_id)
    assert record.error == diagnostic and record.public_error == GENERIC_JOB_FAILURE
    assert (
        restored.connection()
        .execute("SELECT public_error FROM jobs WHERE job_id=?", (job_id,))
        .fetchone()[0]
        is None
    )
    assert record.to_public_dict()["error"] == GENERIC_JOB_FAILURE
    assert restored.transitions(job_id) == transitions
    assert (
        restored.connection()
        .execute("SELECT value FROM schema_meta WHERE key='schema_version'")
        .fetchone()[0]
        == "9"
    )
    # A matching legacy terminal retry remains a no-op, not a diagnostic rewrite.
    assert restored.transition(job_id, "failed", error=diagnostic) == record
    assert restored.transitions(job_id) == transitions
    restored.close()


def test_terminal_retry_cannot_change_public_projection(tmp_path: Path) -> None:
    """Sealed diagnostics cannot acquire a different public reason on retry."""
    import pytest

    from sc_neurocore.studio.platform.jobs_failures import StudioJobError
    from sc_neurocore.studio.platform.jobs_models import StudioJobRejected

    ledger = StudioJobLedger(root=tmp_path)
    job_id = "sj_0123456789abcdef"
    ledger.create(
        job_id=job_id,
        kind="analysis",
        actor="operator",
        workspace="default",
        request_id=None,
        idempotency_key=None,
        experiment_sha256=None,
        admission=None,
        execution_model="thread",
    )
    error = authored_job_error("Source-owned refusal.")
    record = ledger.transition(job_id, "failed", error=error)
    assert ledger.transition(job_id, "failed", error=error) == record
    with pytest.raises(StudioJobRejected, match="public_error"):
        ledger.transition(
            job_id, "failed", error=StudioJobError(str(error), public_message="Different reason.")
        )
    assert ledger.record(job_id) == record
    ledger.close()
