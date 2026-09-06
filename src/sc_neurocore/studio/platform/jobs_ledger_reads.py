# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio job ledger reads

"""Reading the ledger: one job, a scoped list, a history, the live rows.

Reads take no transaction — SQLite in WAL mode gives a reader a consistent
snapshot without blocking the writer. Scoping by actor and workspace happens
here rather than in the caller, so a job outside the scope raises ``KeyError``
instead of being filtered out somewhere that might forget.
"""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING, Any

from sc_neurocore.studio.platform.jobs_ledger_schema import LIVE_STATUSES, record_from_row
from sc_neurocore.studio.platform.jobs_models import StudioJobRecord

if TYPE_CHECKING:  # pragma: no cover - import cycle only matters to type checkers
    from sc_neurocore.studio.platform.jobs_ledger import StudioJobLedger


def read_record(
    ledger: StudioJobLedger,
    job_id: str,
    *,
    actor: str | None = None,
    workspace: str | None = None,
) -> StudioJobRecord:
    """Return one job record, scoped to an actor and workspace when given.

    A job that exists but belongs to a different actor or workspace raises
    :class:`KeyError`, so an isolated caller cannot tell it apart from a job
    that never existed.
    """
    row = ledger.connection().execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
    if row is None:
        raise KeyError(job_id)
    if actor is not None and str(row["actor"]) != actor:
        raise KeyError(job_id)
    if workspace is not None and str(row["workspace"]) != workspace:
        raise KeyError(job_id)
    return record_from_row(row)


def read_records(
    ledger: StudioJobLedger, *, actor: str | None = None, workspace: str | None = None
) -> tuple[StudioJobRecord, ...]:
    """Return records in creation order, scoped to an actor and workspace.

    Creation timestamps have one-second resolution, so two jobs submitted in
    the same second tie. The tie is broken by insertion order (``rowid``), not
    by ``job_id``: an id is a digest, and ordering by it made "the latest job"
    a matter of which random hex sorted higher. Retention decisions read this
    order, so the wrong archive was kept whenever the ids happened to sort
    against submission order.
    """
    rows = (
        ledger.connection()
        .execute(
            "SELECT * FROM jobs WHERE (? IS NULL OR actor = ?)"
            " AND (? IS NULL OR workspace = ?) ORDER BY created_at_utc, rowid",
            (actor, actor, workspace, workspace),
        )
        .fetchall()
    )
    return tuple(record_from_row(row) for row in rows)


def read_transitions(ledger: StudioJobLedger, job_id: str) -> tuple[dict[str, Any], ...]:
    """Return the append-only transition history of one job, in order."""
    rows = (
        ledger.connection()
        .execute("SELECT * FROM job_transitions WHERE job_id = ? ORDER BY sequence", (job_id,))
        .fetchall()
    )
    return tuple(
        {
            "sequence": int(row["sequence"]),
            "from_status": None if row["from_status"] is None else str(row["from_status"]),
            "to_status": str(row["to_status"]),
            "at_utc": str(row["at_utc"]),
            "actor": str(row["actor"]),
            "reason": None if row["reason"] is None else str(row["reason"]),
        }
        for row in rows
    )


def read_live_rows(ledger: StudioJobLedger) -> tuple[sqlite3.Row, ...]:
    """Return the stored rows of every job that has not finished."""
    return tuple(
        ledger.connection()
        .execute("SELECT * FROM jobs WHERE status IN (?, ?, ?, ?)", LIVE_STATUSES)
        .fetchall()
    )


__all__ = ["read_live_rows", "read_record", "read_records", "read_transitions"]
