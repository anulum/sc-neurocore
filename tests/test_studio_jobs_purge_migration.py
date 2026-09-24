# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Purge journal migration custody

"""Forward journal migration preserves every preceding schema's retained evidence."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from sc_neurocore.studio.platform import jobs_admission_schema
from sc_neurocore.studio.platform.jobs_ledger import StudioJobLedger
from sc_neurocore.studio.platform.jobs_ledger_schema import (
    JOB_LEDGER_SCHEMA_VERSION,
    LEDGER_FILENAME,
    SCHEMA_V1,
    SCHEMA_VERSION,
)


@pytest.mark.parametrize("version", [1, 2, 3, 4])
@pytest.mark.parametrize("interrupted", [False, True])
def test_purge_migration_preserves_old_custody_and_rolls_back(
    tmp_path: Path, version: int, interrupted: bool
) -> None:
    """A failure after journal creation leaves an exact old database dump and retry path."""
    path = tmp_path / LEDGER_FILENAME
    tables = ["jobs", "job_transitions"]
    with sqlite3.connect(path) as connection:
        connection.executescript(SCHEMA_V1)
        connection.execute("INSERT INTO schema_meta VALUES('schema_version',?)", (str(version),))
        connection.execute(
            "INSERT INTO schema_meta VALUES('schema_name',?)", (f"studio.job-ledger.v{version}",)
        )
        connection.execute(
            "INSERT INTO jobs(job_id,kind,actor,workspace,admission,execution_model,status,"
            "created_at_utc,artifacts,sequence) VALUES('sj_0000000000000001','analysis',"
            "'owner','default','{}','process','completed','2026-09-08T00:00:00Z','[]',0)"
        )
        if version >= 2:
            jobs_admission_schema.migrate_admission(connection)
            connection.execute("INSERT INTO admission_config VALUES(1,2,3,17,11)")
            tables.extend(["admission_config", "admission_reservations"])
        if version >= 3:
            jobs_admission_schema.migrate_worker_custody(connection)
            connection.execute(
                "INSERT INTO job_workers VALUES('sj_0000000000000001','retained-supervisor',"
                "'retained-worker','retained-boot',123)"
            )
            tables.append("job_workers")
        if version >= 4:
            jobs_admission_schema.migrate_purge_journal(connection)
            connection.executemany(
                "INSERT INTO job_purges VALUES(?,?,?,?,?)",
                [
                    ("legacy-prepared", "owner-a", 10, 20, "prepared"),
                    ("legacy-committed", "owner-b", 30, 40, "committed"),
                ],
            )
            tables.append("job_purges")
        before = {
            table: connection.execute(f"SELECT * FROM {table}").fetchall() for table in tables
        }
        if interrupted:
            # A reviewer's trigger refuses the final version bump, after every
            # migration step already ran inside the same transaction.
            connection.execute(
                "CREATE TRIGGER hold_schema_version BEFORE UPDATE ON schema_meta "
                "BEGIN SELECT RAISE(ABORT, 'schema version held for review'); END"
            )
        connection.commit()
        dump = list(connection.iterdump())
    if interrupted:
        with pytest.raises(sqlite3.IntegrityError, match="schema version held for review"):
            StudioJobLedger(root=tmp_path)
        with sqlite3.connect(path) as connection:
            assert list(connection.iterdump()) == dump
            connection.execute("DROP TRIGGER hold_schema_version")
    ledger = StudioJobLedger(root=tmp_path)
    try:
        for table in tables:
            assert [
                tuple(row) for row in ledger.connection().execute(f"SELECT * FROM {table}")
            ] == before[table]
        assert ledger.connection().execute("SELECT COUNT(*) FROM job_purges").fetchone()[0] == (
            2 if version == 4 else 0
        )
        assert dict(ledger.connection().execute("SELECT key,value FROM schema_meta")) == {
            "schema_version": str(SCHEMA_VERSION),
            "schema_name": JOB_LEDGER_SCHEMA_VERSION,
        }
    finally:
        ledger.close()


@pytest.mark.parametrize("custom", ["index", "trigger"])
def test_phase_migration_preserves_unrecognised_schema_objects(tmp_path: Path, custom: str) -> None:
    """Refused automatic migration leaves custom custody logic and the old dump intact."""
    path = tmp_path / LEDGER_FILENAME
    with sqlite3.connect(path) as connection:
        connection.executescript(SCHEMA_V1)
        connection.execute("INSERT INTO schema_meta VALUES('schema_version','4')")
        jobs_admission_schema.migrate_admission(connection)
        jobs_admission_schema.migrate_worker_custody(connection)
        jobs_admission_schema.migrate_purge_journal(connection)
        if custom == "index":
            connection.execute("CREATE INDEX custom_custody ON job_purges(inode)")
        else:
            connection.execute(
                "CREATE TRIGGER custom_custody BEFORE DELETE ON job_purges "
                "BEGIN SELECT RAISE(ABORT,'custom custody'); END"
            )
        connection.commit()
        before = list(connection.iterdump())
    with pytest.raises(sqlite3.IntegrityError, match="custom schema objects"):
        StudioJobLedger(root=tmp_path)
    with sqlite3.connect(path) as connection:
        assert list(connection.iterdump()) == before
