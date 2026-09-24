# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — duplicate requests waiting in the shared admission queue

"""Two identical requests queued at once resolve to one job, never two.

A request and its retry (a lost reply, or the same idempotency key) can both be
waiting for capacity. Real threads queue both behind a real running job; when
the slot is released one is admitted and the other, on its next queue pass,
returns that admission and gives up its own queued reservation.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time

import pytest

from sc_neurocore.studio.platform.jobs_admission_replay import StorageAdmissionReplay
from sc_neurocore.studio.platform.jobs_ledger import StudioJobLedger
from sc_neurocore.studio.platform.jobs_ledger_schema import StudioJobSubmission
from sc_neurocore.studio.platform.jobs_shared_admission import SharedJobAdmission


def _submit(admission: SharedJobAdmission, job_id: str, key: str) -> StudioJobSubmission:
    return admission.admit(
        job_id=job_id,
        kind="analysis",
        actor="studio-service",
        workspace="default",
        request_id=None,
        idempotency_key="same-request" if key == "idempotency" else None,
        experiment_sha256=None,
        admission=None,
        execution_model="process",
        replay=(
            StorageAdmissionReplay("operator", "same-request", "a" * 64)
            if key == "replay"
            else None
        ),
    )


@pytest.mark.parametrize("key", ["replay", "idempotency"])
def test_queued_duplicate_returns_the_admitted_request(tmp_path: Path, key: str) -> None:
    """The second queued copy returns the first admission and leaves the queue."""
    ledger = StudioJobLedger(root=tmp_path)
    admission = SharedJobAdmission(ledger, max_concurrent=1, max_queued=2)
    try:
        occupying = _submit(admission, "sj_0000000000000001", "none")
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(_submit, admission, f"sj_000000000000000{index}", key)
                for index in (2, 3)
            ]
            deadline = time.monotonic() + 10.0
            while admission.snapshot().queued < 2:
                assert time.monotonic() < deadline, "both copies never queued"
                time.sleep(0.01)
            # The running job finishes and its supervisor frees the slot.
            ledger.transition(occupying.record.job_id, "running")
            ledger.transition(occupying.record.job_id, "completed")
            admission.release(job_id=occupying.record.job_id)
            outcomes = [future.result(timeout=30.0) for future in futures]
        admitted = [outcome for outcome in outcomes if not outcome.duplicate]
        duplicates = [outcome for outcome in outcomes if outcome.duplicate]
        assert len(admitted) == 1 and len(duplicates) == 1
        assert duplicates[0].record.job_id == admitted[0].record.job_id
        assert admission.snapshot().queued == 0
        assert admission.snapshot().running == 1
        assert len(ledger.list_records()) == 2
    finally:
        ledger.close()


def test_empty_delegated_supervisor_is_refused(tmp_path: Path) -> None:
    """A delegated admission must name the supervisor it delegates to."""
    ledger = StudioJobLedger(root=tmp_path)
    admission = SharedJobAdmission(ledger, max_concurrent=1, max_queued=0)
    try:
        with pytest.raises(ValueError, match="must be nonempty"):
            admission.admit(
                job_id="sj_0000000000000001",
                kind="analysis",
                actor="studio-service",
                workspace="default",
                request_id=None,
                idempotency_key=None,
                experiment_sha256=None,
                admission=None,
                execution_model="process",
                supervisor="",
            )
        assert ledger.list_records() == ()
    finally:
        ledger.close()
