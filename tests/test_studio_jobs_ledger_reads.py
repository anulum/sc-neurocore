# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Job ledger query contracts

"""Query the durable ledger through its public filtering and recovery API."""

from __future__ import annotations

from pathlib import Path

import pytest

from sc_neurocore.studio.platform.jobs_ledger import StudioJobLedger
from sc_neurocore.studio.platform.jobs_ledger_schema import ALLOWED_TRANSITIONS, LIVE_STATUSES


def _admit(ledger: StudioJobLedger, index: int, actor: str, workspace: str) -> str:
    job_id = f"sj_{index:016x}"
    ledger.create(
        job_id=job_id,
        kind="analysis",
        actor=actor,
        workspace=workspace,
        request_id=None,
        idempotency_key=None,
        experiment_sha256=None,
        admission=None,
        execution_model="thread",
    )
    return job_id


@pytest.mark.parametrize(
    ("actor", "workspace", "expected"),
    [
        (None, None, (1, 2, 3)),
        ("alice", None, (1, 2)),
        (None, "lab-a", (1, 3)),
        ("alice", "lab-a", (1,)),
        ("alice", "missing", ()),
        ("", None, ()),
    ],
)
def test_optional_filters_intersect_and_preserve_order(
    tmp_path: Path, actor: str | None, workspace: str | None, expected: tuple[int, ...]
) -> None:
    """Optional scope filters retain exact rows, not merely a matching count."""
    ledger = StudioJobLedger(root=tmp_path)
    try:
        for index, (owner, space) in enumerate(
            [("alice", "lab-a"), ("alice", "lab-b"), ("bob", "lab-a")], start=1
        ):
            _admit(ledger, index, owner, space)
        rows = ledger.list_records(actor=actor, workspace=workspace)
        assert tuple(row.job_id for row in rows) == tuple(f"sj_{i:016x}" for i in expected)
    finally:
        ledger.close()


def test_live_query_covers_every_nonterminal_state(tmp_path: Path) -> None:
    """Recovery sees all live states and none of the terminal records."""
    ledger = StudioJobLedger(root=tmp_path)
    expected: set[str] = set()
    try:
        for index, status in enumerate(ALLOWED_TRANSITIONS, start=1):
            job_id = _admit(ledger, index, "alice", "lab-a")
            if status == "completed":
                ledger.transition(job_id, "running")
            if status != "pending":
                ledger.transition(job_id, status)
            if status in LIVE_STATUSES:
                expected.add(job_id)
        assert {str(row["job_id"]) for row in ledger.live_rows()} == expected
    finally:
        ledger.close()
