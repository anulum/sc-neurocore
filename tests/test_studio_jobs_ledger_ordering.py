# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio job ledger ordering

"""The order jobs come back in decides which one retention keeps.

Creation timestamps have one-second resolution, so jobs submitted together
tie. What breaks the tie is the whole question: an id is a digest, and
ordering by it makes "the latest job" a matter of which random hex sorts
higher.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from sc_neurocore.studio.platform.jobs_ledger import StudioJobLedger

UTC_CLOCK_START = datetime.fromisoformat("2026-09-06T00:00:00+00:00")


def _ledger(root: Path, **kwargs: object) -> StudioJobLedger:
    return StudioJobLedger(root=root, **kwargs)  # type: ignore[arg-type]


def _admit(ledger: StudioJobLedger, job_id: str) -> object:
    return ledger.create(
        job_id=job_id,
        kind="analysis",
        actor="alice",
        workspace="default",
        request_id=None,
        idempotency_key=None,
        experiment_sha256=None,
        admission=None,
        execution_model="thread",
    )


class TestOrdering:
    def test_jobs_created_in_the_same_second_come_back_in_submission_order(
        self, tmp_path: Path
    ) -> None:
        """Retention reads this order, so a tie must not be decided by a digest.

        Creation timestamps have one-second resolution. Ordering ties by
        ``job_id`` meant "the latest job" was whichever random hex sorted
        higher, so "retain the latest archive" kept the wrong one whenever the
        ids happened to sort against submission order.
        """
        ledger = _ledger(tmp_path, clock=lambda: UTC_CLOCK_START)
        submitted = ["sj_ffffffffffffffff", "sj_0000000000000001", "sj_aaaaaaaaaaaaaaaa"]
        for job_id in submitted:
            _admit(ledger, job_id=job_id)

        assert [record.job_id for record in ledger.list_records()] == submitted

    def test_the_order_survives_reopening_the_ledger(self, tmp_path: Path) -> None:
        ledger = _ledger(tmp_path, clock=lambda: UTC_CLOCK_START)
        submitted = ["sj_ffffffffffffffff", "sj_0000000000000001"]
        for job_id in submitted:
            _admit(ledger, job_id=job_id)

        assert [record.job_id for record in _ledger(tmp_path).list_records()] == submitted
