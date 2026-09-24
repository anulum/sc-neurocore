# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — durable storage admission replay

"""Persist an isolated storage mutation's exact outcome with job admission."""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass

from sc_neurocore.studio.platform.jobs_admission import StudioJobQueueFull
from sc_neurocore.studio.platform.jobs_ledger_schema import StudioJobSubmission
from sc_neurocore.studio.platform.jobs_models import StudioJobRejected
from sc_neurocore.studio.platform.jobs_snapshot import decode_job_snapshot

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True, slots=True)
class StorageAdmissionReplay:
    """One authenticated requester, server workspace and exact mutation identity.

    The trusted storage service computes ``payload_sha256`` from its validated
    versioned request. A browser trace ID and the legacy job idempotency key do
    not replace this identity.
    """

    requester: str
    mutation_id: str
    payload_sha256: str

    def validate(self) -> None:
        """Reject malformed replay identities before any reservation is opened."""
        if (
            not isinstance(self.requester, str)
            or not isinstance(self.mutation_id, str)
            or not isinstance(self.payload_sha256, str)
            or not self.requester
            or not self.mutation_id
            or self.requester != self.requester.strip()
            or self.mutation_id != self.mutation_id.strip()
            or not self.requester.isprintable()
            or not self.mutation_id.isprintable()
            or len(self.requester) > 256
            or len(self.mutation_id) > 256
            or _SHA256.fullmatch(self.payload_sha256) is None
        ):
            raise ValueError("invalid storage admission replay identity")


def read_admission_replay(
    connection: sqlite3.Connection,
    *,
    workspace: str,
    replay: StorageAdmissionReplay,
) -> StudioJobSubmission | StudioJobQueueFull | None:
    """Return an immutable prior outcome or reject a changed request digest."""
    row = connection.execute(
        "SELECT payload_sha256,outcome,record_json,running,queued,queue_limit "
        "FROM storage_admission_replays WHERE workspace=? AND requester=? AND mutation_id=?",
        (workspace, replay.requester, replay.mutation_id),
    ).fetchone()
    if row is None:
        return None
    if row["payload_sha256"] != replay.payload_sha256:
        raise StudioJobRejected("Storage mutation identity was reused with changed content.")
    if row["outcome"] == "admitted" and row["record_json"] is not None:
        try:
            record = decode_job_snapshot(json.loads(str(row["record_json"])))
        except (ValueError, TypeError) as exc:
            raise StudioJobRejected("Stored admission replay is invalid.") from exc
        if record.workspace != workspace:
            raise StudioJobRejected("Stored admission replay workspace is invalid.")
        return StudioJobSubmission(record, duplicate=True)
    if row["outcome"] == "refused" and all(
        row[name] is not None for name in ("running", "queued", "queue_limit")
    ):
        return StudioJobQueueFull(
            running=int(row["running"]),
            queued=int(row["queued"]),
            limit=int(row["queue_limit"]),
        )
    raise StudioJobRejected("Stored admission replay is invalid.")


def write_admission_replay(
    connection: sqlite3.Connection,
    *,
    workspace: str,
    replay: StorageAdmissionReplay,
    outcome: StudioJobSubmission | StudioJobQueueFull,
) -> None:
    """Write the exact result inside the caller's job/capacity transaction."""
    if isinstance(outcome, StudioJobSubmission):
        fields: tuple[object, ...] = (
            "admitted",
            json.dumps(outcome.record.to_public_dict(), sort_keys=True, allow_nan=False),
            None,
            None,
            None,
        )
    else:
        fields = ("refused", None, outcome.running, outcome.queued, outcome.limit)
    connection.execute(
        "INSERT INTO storage_admission_replays "
        "(workspace,requester,mutation_id,payload_sha256,outcome,record_json,"
        "running,queued,queue_limit) VALUES(?,?,?,?,?,?,?,?,?)",
        (workspace, replay.requester, replay.mutation_id, replay.payload_sha256, *fields),
    )
