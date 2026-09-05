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

import json
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from sc_neurocore.studio.platform.jobs_models import (
    StudioJobArtifact,
    StudioJobRecord,
    StudioJobStatus,
)

JOB_LEDGER_SCHEMA_VERSION = "studio.job-ledger.v1"
LEDGER_FILENAME = "job_ledger.sqlite3"
SCHEMA_VERSION = 1

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


class StudioJobLedgerCorrupt(RuntimeError):
    """Raised when the ledger file cannot be read as a Studio job ledger."""


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
    # Forward migrations land here as `if stored < N:` steps. Version 1 is the
    # initial schema, which SCHEMA_V1 already ensures.


def json_or_none(value: str | None) -> Any:
    """Decode one stored JSON column, or ``None``."""
    if value is None:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise StudioJobLedgerCorrupt(f"stored JSON column is not valid JSON: {exc}") from exc


def artifacts_from_json(value: str) -> tuple[StudioJobArtifact, ...]:
    """Rebuild an artifact manifest, refusing a malformed one."""
    payload = json_or_none(value) or []
    if not isinstance(payload, list):
        raise StudioJobLedgerCorrupt("the artifact manifest is not a list")
    artifacts: list[StudioJobArtifact] = []
    for entry in payload:
        if not isinstance(entry, Mapping):
            raise StudioJobLedgerCorrupt("an artifact manifest entry is not an object")
        try:
            artifacts.append(
                StudioJobArtifact(
                    relative_path=str(entry["relative_path"]),
                    size_bytes=int(entry["size_bytes"]),
                    sha256=str(entry["sha256"]),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise StudioJobLedgerCorrupt(f"an artifact manifest entry is malformed: {exc}") from exc
    return tuple(artifacts)


def artifacts_to_json(artifacts: Sequence[StudioJobArtifact]) -> str:
    """Serialise an artifact manifest deterministically."""
    return json.dumps([artifact.to_public_dict() for artifact in artifacts], sort_keys=True)


def record_from_row(row: sqlite3.Row) -> StudioJobRecord:
    """Rebuild one immutable public record from its stored row."""
    return StudioJobRecord(
        job_id=str(row["job_id"]),
        kind=str(row["kind"]),
        owner=str(row["actor"]),
        request_id=None if row["request_id"] is None else str(row["request_id"]),
        status=str(row["status"]),  # type: ignore[arg-type]
        execution_model=str(row["execution_model"]),  # type: ignore[arg-type]
        created_at_utc=str(row["created_at_utc"]),
        started_at_utc=None if row["started_at_utc"] is None else str(row["started_at_utc"]),
        finished_at_utc=None if row["finished_at_utc"] is None else str(row["finished_at_utc"]),
        error=None if row["error"] is None else str(row["error"]),
        result=json_or_none(row["result"]),
        artifacts=artifacts_from_json(str(row["artifacts"])),
        workspace=str(row["workspace"]),
        idempotency_key=None if row["idempotency_key"] is None else str(row["idempotency_key"]),
        experiment_sha256=(
            None if row["experiment_sha256"] is None else str(row["experiment_sha256"])
        ),
        admission=json_or_none(row["admission"]) or {},
        lease_owner=None if row["lease_owner"] is None else str(row["lease_owner"]),
        lease_expires_at_utc=(
            None if row["lease_expires_at_utc"] is None else str(row["lease_expires_at_utc"])
        ),
        heartbeat_at_utc=None if row["heartbeat_at_utc"] is None else str(row["heartbeat_at_utc"]),
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
