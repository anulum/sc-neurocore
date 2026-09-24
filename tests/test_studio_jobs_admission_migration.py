# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Capacity migration preservation

"""Open a genuine version-one SQLite ledger through the production migrator."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from sc_neurocore.studio.platform.jobs_ledger import StudioJobLedger
from sc_neurocore.studio.platform.jobs_ledger_schema import (
    LEDGER_FILENAME,
    SCHEMA_V1,
    SCHEMA_VERSION,
)


@pytest.mark.parametrize("prior_version", [1, 2])
def test_worker_migration_failure_retains_previous_schema(
    tmp_path: Path, prior_version: int
) -> None:
    """Failure after worker-table creation rolls back the full forward migration."""
    from sc_neurocore.studio.platform import jobs_admission_schema

    path = tmp_path / LEDGER_FILENAME
    with sqlite3.connect(path) as connection:
        connection.executescript(SCHEMA_V1)
        if prior_version == 2:
            jobs_admission_schema.migrate_admission(connection)
        connection.execute(
            "INSERT INTO schema_meta VALUES('schema_version',?)", (str(prior_version),)
        )
        # A reviewer's trigger refuses the final version bump, after the worker
        # table was created inside the same migration transaction.
        connection.execute(
            "CREATE TRIGGER hold_schema_version BEFORE UPDATE ON schema_meta "
            "BEGIN SELECT RAISE(ABORT, 'schema version held for review'); END"
        )
        before = connection.execute("SELECT name,sql FROM sqlite_master ORDER BY name").fetchall()
    with pytest.raises(sqlite3.IntegrityError, match="schema version held for review"):
        StudioJobLedger(root=tmp_path)
    with sqlite3.connect(path) as connection:
        assert (
            connection.execute("SELECT name,sql FROM sqlite_master ORDER BY name").fetchall()
            == before
        )
        assert connection.execute(
            "SELECT value FROM schema_meta WHERE key='schema_version'"
        ).fetchone()[0] == str(prior_version)
        connection.execute("DROP TRIGGER hold_schema_version")
    retry = StudioJobLedger(root=tmp_path)
    try:
        assert retry.connection().execute("SELECT COUNT(*) FROM job_workers").fetchone()[0] == 0
        assert retry.connection().execute(
            "SELECT value FROM schema_meta WHERE key='schema_version'"
        ).fetchone()[0] == str(SCHEMA_VERSION)
    finally:
        retry.close()


def test_worker_identity_migration_preserves_existing_shared_capacity(tmp_path: Path) -> None:
    """A version-two root gains custody storage without fabricating worker evidence."""
    from sc_neurocore.studio.platform.jobs_admission_schema import migrate_admission

    with sqlite3.connect(tmp_path / LEDGER_FILENAME) as connection:
        connection.executescript(SCHEMA_V1)
        migrate_admission(connection)
        connection.execute("INSERT INTO schema_meta VALUES('schema_version','2')")
        connection.execute("INSERT INTO schema_meta VALUES('schema_name','studio.job-ledger.v2')")
        connection.execute("INSERT INTO admission_config VALUES(1,2,3,5,7)")
        connection.execute(
            "INSERT INTO admission_reservations(job_id,supervisor,state) VALUES('legacy','owner','unreaped')"
        )
    ledger = StudioJobLedger(root=tmp_path)
    try:
        assert tuple(ledger.connection().execute("SELECT * FROM admission_config").fetchone()) == (
            1,
            2,
            3,
            5,
            7,
        )
        assert (
            ledger.connection().execute("SELECT state FROM admission_reservations").fetchone()[0]
            == "unreaped"
        )
        assert ledger.connection().execute("SELECT COUNT(*) FROM job_workers").fetchone()[0] == 0
        assert ledger.connection().execute(
            "SELECT value FROM schema_meta WHERE key='schema_version'"
        ).fetchone()[0] == str(SCHEMA_VERSION)
    finally:
        ledger.close()


def test_interrupted_migration_rolls_back_before_retry(tmp_path: Path) -> None:
    """A failed capacity migration leaves the old schema version and no half-tables."""

    path = tmp_path / LEDGER_FILENAME
    with sqlite3.connect(path) as connection:
        connection.executescript(SCHEMA_V1)
        connection.execute("INSERT INTO schema_meta VALUES ('schema_version','1')")
        # The capacity tables are created before the refused version bump.
        connection.execute(
            "CREATE TRIGGER hold_schema_version BEFORE UPDATE ON schema_meta "
            "BEGIN SELECT RAISE(ABORT, 'schema version held for review'); END"
        )
    with pytest.raises(sqlite3.IntegrityError, match="schema version held for review"):
        StudioJobLedger(root=tmp_path)
    with sqlite3.connect(path) as connection:
        assert (
            connection.execute(
                "SELECT value FROM schema_meta WHERE key='schema_version'"
            ).fetchone()[0]
            == "1"
        )
        assert (
            connection.execute(
                "SELECT name FROM sqlite_master WHERE name LIKE 'admission_%'"
            ).fetchall()
            == []
        )
        connection.execute("DROP TRIGGER hold_schema_version")
    retried = StudioJobLedger(root=tmp_path)
    assert retried.connection().execute(
        "SELECT value FROM schema_meta WHERE key='schema_version'"
    ).fetchone()[0] == str(SCHEMA_VERSION)
    retried.close()


@pytest.mark.parametrize(
    "status,error,expected",
    [
        ("pending", None, "running"),
        ("running", None, "running"),
        ("cancelling", None, "running"),
        ("unknown", None, "running"),
        ("completed", None, None),
        ("timed_out", "worker did not stop", "unreaped"),
        ("failed", "Worker stopped: False.", "unreaped"),
        ("cancelled", "The worker process group was not reaped", "unreaped"),
    ],
)
def test_migration_preserves_legacy_capacity_and_job_rows(
    tmp_path: Path, status: str, error: str | None, expected: str | None
) -> None:
    """Migration retains live/unknown/unreaped capacity and never edits job evidence."""
    path = tmp_path / LEDGER_FILENAME
    with sqlite3.connect(path) as connection:
        connection.executescript(SCHEMA_V1)
        connection.execute("INSERT INTO schema_meta VALUES ('schema_version','1')")
        connection.execute("INSERT INTO schema_meta VALUES ('schema_name','studio.job-ledger.v1')")
        connection.execute(
            "INSERT INTO jobs(job_id,kind,actor,workspace,admission,execution_model,status,"
            "created_at_utc,error,artifacts,lease_owner,sequence) "
            "VALUES('sj_0000000000000001','analysis','owner','default','{}','thread',?,"
            "'2026-09-08T00:00:00Z',?,'[]','legacy-supervisor',0)",
            (status, error),
        )
        before = connection.execute("SELECT * FROM jobs").fetchall()
    ledger = StudioJobLedger(root=tmp_path)
    rows = (
        ledger.connection()
        .execute("SELECT job_id,supervisor,state FROM admission_reservations")
        .fetchall()
    )
    assert [tuple(row) for row in rows] == (
        [] if expected is None else [("sj_0000000000000001", "legacy-supervisor", expected)]
    )
    assert [tuple(row) for row in ledger.connection().execute("SELECT * FROM jobs")] == before
    assert ledger.connection().execute("SELECT COUNT(*) FROM admission_config").fetchone()[0] == 0
    assert ledger.connection().execute(
        "SELECT value FROM schema_meta WHERE key='schema_version'"
    ).fetchone()[0] == str(SCHEMA_VERSION)
    ledger.close()
    reopened = StudioJobLedger(root=tmp_path)
    assert [
        tuple(row)
        for row in reopened.connection().execute(
            "SELECT job_id,supervisor,state FROM admission_reservations"
        )
    ] == [tuple(row) for row in rows]
    reopened.close()
