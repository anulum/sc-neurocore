# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Shared job capacity schema

"""Transactional reservation tables inside the existing job ledger."""

from __future__ import annotations

import sqlite3


def migrate_purge_journal(connection: sqlite3.Connection) -> None:
    """Record exact directory custody before a filesystem/database purge begins."""
    connection.execute(
        "CREATE TABLE IF NOT EXISTS job_purges ("
        "job_id TEXT PRIMARY KEY, supervisor TEXT NOT NULL, device INTEGER, inode INTEGER, "
        "state TEXT NOT NULL CHECK(state IN ('prepared','committed')))"
    )


def migrate_purge_phases(connection: sqlite3.Connection) -> None:
    """Extend purge phases without inferring completion evidence for legacy rows.

    The caller owns the transaction. Refuse custom indexes/triggers rather than
    silently destroying them while replacing the constrained table definition.
    """
    custom = connection.execute(
        "SELECT name FROM sqlite_master WHERE tbl_name='job_purges' "
        "AND type IN ('index','trigger') AND sql IS NOT NULL"
    ).fetchall()
    if custom:
        raise sqlite3.IntegrityError("Purge migration requires review of custom schema objects.")
    connection.execute(
        "CREATE TABLE job_purges_v5 ("
        "job_id TEXT PRIMARY KEY, supervisor TEXT NOT NULL, device INTEGER, inode INTEGER, "
        "state TEXT NOT NULL CHECK(state IN "
        "('prepared','committed','cleanup_started','removed','ambiguous')))"
    )
    connection.execute("INSERT INTO job_purges_v5 SELECT * FROM job_purges")
    connection.execute("DROP TABLE job_purges")
    connection.execute("ALTER TABLE job_purges_v5 RENAME TO job_purges")


def migrate_worker_custody(connection: sqlite3.Connection) -> None:
    """Add worker identity evidence without inferring identities for historical jobs."""
    connection.execute(
        "CREATE TABLE IF NOT EXISTS job_workers ("
        "job_id TEXT PRIMARY KEY, supervisor TEXT NOT NULL, "
        "worker_identity TEXT NOT NULL, boot_id TEXT NOT NULL, "
        "group_id INTEGER NOT NULL CHECK(group_id>0))"
    )


def migrate_admission(connection: sqlite3.Connection) -> None:
    """Create shared capacity tables and retain legacy occupied capacity.

    The caller owns the migration transaction. Do not use executescript here:
    it would commit the surrounding migration early. Job records and transition
    history are not rewritten. Unknown jobs and explicitly unreaped outcomes
    retain a reservation; their capacity cannot be reclaimed from expiry alone.
    """
    connection.execute(
        "CREATE TABLE IF NOT EXISTS admission_config ("
        "singleton INTEGER PRIMARY KEY CHECK(singleton = 1), "
        "max_concurrent INTEGER NOT NULL CHECK(max_concurrent > 0), "
        "max_queued INTEGER NOT NULL CHECK(max_queued >= 0), "
        "admitted INTEGER NOT NULL DEFAULT 0, refused INTEGER NOT NULL DEFAULT 0)"
    )
    connection.execute(
        "CREATE TABLE IF NOT EXISTS admission_reservations ("
        "ticket INTEGER PRIMARY KEY AUTOINCREMENT, "
        "job_id TEXT NOT NULL UNIQUE, supervisor TEXT, "
        "state TEXT NOT NULL CHECK(state IN ('queued', 'running', 'unreaped')))"
    )
    connection.execute(
        "INSERT OR IGNORE INTO admission_reservations(job_id, supervisor, state) "
        "SELECT job_id, lease_owner, CASE "
        "WHEN status IN ('pending','running','cancelling','unknown') THEN 'running' "
        "ELSE 'unreaped' END FROM jobs WHERE "
        "status IN ('pending','running','cancelling','unknown') "
        "OR lower(COALESCE(error,'')) LIKE '%worker did not stop%' "
        "OR lower(COALESCE(error,'')) LIKE '%worker stopped: false%' "
        "OR lower(COALESCE(error,'')) LIKE '%not reaped%'"
    )


def migrate_storage_admission_replay(connection: sqlite3.Connection) -> None:
    """Add immutable exact-outcome replay without inferring legacy request digests."""
    connection.execute(
        "CREATE TABLE IF NOT EXISTS storage_admission_replays ("
        "workspace TEXT NOT NULL, requester TEXT NOT NULL, mutation_id TEXT NOT NULL, "
        "payload_sha256 TEXT NOT NULL, outcome TEXT NOT NULL "
        "CHECK(outcome IN ('admitted','refused')), record_json TEXT, "
        "running INTEGER, queued INTEGER, queue_limit INTEGER, "
        "PRIMARY KEY(workspace,requester,mutation_id), "
        "CHECK((outcome='admitted' AND record_json IS NOT NULL "
        "AND running IS NULL AND queued IS NULL AND queue_limit IS NULL) OR "
        "(outcome='refused' AND record_json IS NULL AND running IS NOT NULL "
        "AND queued IS NOT NULL AND queue_limit IS NOT NULL "
        "AND running>=0 AND queued>=0 AND queue_limit>=0)))"
    )
