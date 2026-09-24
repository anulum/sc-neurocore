# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio job ledger schema and state machine

"""The stored shape of the job ledger, and the moves it permits.

The transition log is append-only because triggers refuse to update or delete
its rows, not because callers are asked nicely. The state machine is explicit
so that a bug which tries to complete an interrupted job fails loudly instead
of quietly rewriting what happened.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from sc_neurocore.studio.platform.jobs_ledger_rows import (
    StudioJobLedgerCorrupt,
    artifacts_from_json,
    artifacts_to_json,
    json_or_none,
    record_from_row,
)
from sc_neurocore.studio.platform.jobs_models import (
    StudioJobRecord,
    StudioJobStatus,
)

JOB_LEDGER_SCHEMA_VERSION = "studio.job-ledger.v7"
LEDGER_FILENAME = "job_ledger.sqlite3"
SCHEMA_VERSION = 7

#: A job in one of these states has finished and will never move again.
TERMINAL_STATUSES: frozenset[StudioJobStatus] = frozenset(
    {"completed", "failed", "cancelled", "timed_out", "interrupted"}
)

#: States a job can still leave. Recovery examines exactly these.
LIVE_STATUSES: tuple[StudioJobStatus, ...] = ("pending", "running", "cancelling", "unknown")

#: The transitions the ledger accepts; anything else is refused.
#:
#: Every live state can reach ``interrupted``: recovery reaches a job at
#: whatever point its supervisor died, and refusing to record that would leave
#: the job stuck in a state nothing will ever leave. A completed job, by
#: contrast, leaves nowhere at all.
ALLOWED_TRANSITIONS: dict[StudioJobStatus, frozenset[StudioJobStatus]] = {
    "pending": frozenset(
        {"running", "cancelling", "cancelled", "failed", "timed_out", "unknown", "interrupted"}
    ),
    "running": frozenset(
        {"completed", "failed", "cancelling", "cancelled", "timed_out", "unknown", "interrupted"}
    ),
    "cancelling": frozenset(
        {"cancelled", "completed", "failed", "timed_out", "unknown", "interrupted"}
    ),
    # Verification resolves an unknown job; it never resolves itself.
    "unknown": frozenset({"interrupted", "completed", "failed", "cancelled", "timed_out"}),
    "completed": frozenset(),
    "failed": frozenset(),
    "cancelled": frozenset(),
    "timed_out": frozenset(),
    "interrupted": frozenset(),
}

SCHEMA_V1 = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    actor TEXT NOT NULL,
    workspace TEXT NOT NULL,
    request_id TEXT,
    idempotency_key TEXT,
    experiment_sha256 TEXT,
    admission TEXT NOT NULL,
    training_config TEXT,
    execution_model TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at_utc TEXT NOT NULL,
    started_at_utc TEXT,
    finished_at_utc TEXT,
    error TEXT,
    result TEXT,
    artifacts TEXT NOT NULL,
    lease_owner TEXT,
    lease_expires_at_utc TEXT,
    heartbeat_at_utc TEXT,
    sequence INTEGER NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS jobs_idempotency
    ON jobs (actor, workspace, idempotency_key)
    WHERE idempotency_key IS NOT NULL;

CREATE INDEX IF NOT EXISTS jobs_created ON jobs (created_at_utc, job_id);

CREATE TABLE IF NOT EXISTS job_transitions (
    job_id TEXT NOT NULL,
    sequence INTEGER NOT NULL,
    from_status TEXT,
    to_status TEXT NOT NULL,
    at_utc TEXT NOT NULL,
    actor TEXT NOT NULL,
    reason TEXT,
    PRIMARY KEY (job_id, sequence)
);

CREATE TRIGGER IF NOT EXISTS job_transitions_no_update
BEFORE UPDATE ON job_transitions
BEGIN
    SELECT RAISE(ABORT, 'job transitions are append-only');
END;

CREATE TRIGGER IF NOT EXISTS job_transitions_no_delete
BEFORE DELETE ON job_transitions
WHEN (SELECT COUNT(*) FROM jobs WHERE jobs.job_id = OLD.job_id) > 0
BEGIN
    SELECT RAISE(ABORT, 'job transitions are append-only');
END;
"""


@dataclass(frozen=True, slots=True)
class StudioJobSubmission:
    """The outcome of asking the ledger to admit one job.

    Attributes
    ----------
    record : StudioJobRecord
        The stored record: the new one, or the existing one when the
        idempotency key had already been admitted.
    duplicate : bool
        ``True`` when an earlier submission already owns this idempotency key,
        so the caller must not start a second run.
    """

    record: StudioJobRecord
    duplicate: bool


def migrate(connection: sqlite3.Connection) -> None:
    """Bring the stored schema forward, refusing a version from the future.

    Raises
    ------
    StudioJobLedgerCorrupt
        The file was written by a newer schema than this build understands.
        Downgrading a ledger would silently drop columns, so it is refused.
    """
    row = connection.execute(
        "SELECT value FROM schema_meta WHERE key = 'schema_version'"
    ).fetchone()
    if row is None:
        from sc_neurocore.studio.platform.jobs_admission_schema import migrate_admission

        migrate_admission(connection)
        from sc_neurocore.studio.platform.jobs_admission_schema import migrate_worker_custody

        migrate_worker_custody(connection)
        from sc_neurocore.studio.platform.jobs_admission_schema import migrate_purge_journal

        migrate_purge_journal(connection)
        from sc_neurocore.studio.platform.jobs_admission_schema import migrate_purge_phases

        migrate_purge_phases(connection)
        from sc_neurocore.studio.platform.jobs_admission_schema import (
            migrate_storage_admission_replay,
        )

        migrate_storage_admission_replay(connection)
        connection.execute(
            "INSERT INTO schema_meta (key, value) VALUES ('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        connection.execute(
            "INSERT OR REPLACE INTO schema_meta (key, value) VALUES ('schema_name', ?)",
            (JOB_LEDGER_SCHEMA_VERSION,),
        )
        return
    stored = int(str(row["value"]))
    if stored > SCHEMA_VERSION:
        raise StudioJobLedgerCorrupt(
            f"the ledger was written by schema version {stored}; this build understands "
            f"{SCHEMA_VERSION}. Upgrade the package rather than downgrading the ledger."
        )
    if stored < 2:
        from sc_neurocore.studio.platform.jobs_admission_schema import migrate_admission

        migrate_admission(connection)
    if stored < 3:
        from sc_neurocore.studio.platform.jobs_admission_schema import migrate_worker_custody

        migrate_worker_custody(connection)
    if stored < 4:
        from sc_neurocore.studio.platform.jobs_admission_schema import migrate_purge_journal

        migrate_purge_journal(connection)
    if stored < 5:
        from sc_neurocore.studio.platform.jobs_admission_schema import migrate_purge_phases

        migrate_purge_phases(connection)
    if stored < 6:
        from sc_neurocore.studio.platform.jobs_admission_schema import (
            migrate_storage_admission_replay,
        )

        migrate_storage_admission_replay(connection)
        connection.execute(
            "UPDATE schema_meta SET value = ? WHERE key = 'schema_version'", (str(SCHEMA_VERSION),)
        )
        connection.execute(
            "INSERT OR REPLACE INTO schema_meta(key,value) VALUES('schema_name',?)",
            (JOB_LEDGER_SCHEMA_VERSION,),
        )
    if stored < 7:
        columns = {str(row["name"]) for row in connection.execute("PRAGMA table_info(jobs)")}
        if "training_config" not in columns:
            connection.execute("ALTER TABLE jobs ADD COLUMN training_config TEXT")
        connection.execute(
            "UPDATE schema_meta SET value = ? WHERE key = 'schema_version'", (str(SCHEMA_VERSION),)
        )
        connection.execute(
            "INSERT OR REPLACE INTO schema_meta(key,value) VALUES('schema_name',?)",
            (JOB_LEDGER_SCHEMA_VERSION,),
        )


__all__ = [
    "ALLOWED_TRANSITIONS",
    "JOB_LEDGER_SCHEMA_VERSION",
    "LEDGER_FILENAME",
    "LIVE_STATUSES",
    "SCHEMA_V1",
    "SCHEMA_VERSION",
    "TERMINAL_STATUSES",
    "StudioJobLedgerCorrupt",
    "StudioJobSubmission",
    "artifacts_from_json",
    "artifacts_to_json",
    "json_or_none",
    "migrate",
    "record_from_row",
]
