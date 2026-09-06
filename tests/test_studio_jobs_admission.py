# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio job admission control

"""How many jobs run at once, and what the overflow is told.

The failure these guard against is not a slow Studio: it is a Studio that
accepts everything, starts everything, and then cannot answer anything. A
refusal with a reason is a better answer than an admission that never
completes.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from sc_neurocore.studio.platform.jobs import StudioJobContext, StudioJobManager
from sc_neurocore.studio.platform.jobs_admission import (
    StudioJobAdmission,
    StudioJobQueueFull,
)


class TestAdmissionController:
    def test_slots_are_taken_and_given_back(self) -> None:
        admission = StudioJobAdmission(max_concurrent=2, max_queued=0)

        admission.reserve()
        admission.reserve()
        assert admission.snapshot().running == 2

        with pytest.raises(StudioJobQueueFull):
            admission.reserve()

        admission.release()
        admission.reserve()
        snapshot = admission.snapshot()
        assert snapshot.running == 2
        assert snapshot.admitted == 3
        assert snapshot.refused == 1

    def test_a_waiting_submission_is_admitted_when_a_slot_frees(self) -> None:
        admission = StudioJobAdmission(max_concurrent=1, max_queued=1)
        admission.reserve()
        admitted = threading.Event()

        def waiter() -> None:
            admission.reserve()
            admitted.set()

        thread = threading.Thread(target=waiter, daemon=True)
        thread.start()
        time.sleep(0.05)
        assert admitted.is_set() is False
        assert admission.snapshot().queued == 1

        admission.release()

        assert admitted.wait(30.0) is True
        thread.join(timeout=30.0)

    def test_a_wait_that_times_out_is_refused_and_leaves_the_queue(self) -> None:
        admission = StudioJobAdmission(max_concurrent=1, max_queued=1)
        admission.reserve()

        with pytest.raises(StudioJobQueueFull):
            admission.reserve(timeout_seconds=0.05)

        assert admission.snapshot().queued == 0

    def test_the_refusal_states_why(self) -> None:
        admission = StudioJobAdmission(max_concurrent=1, max_queued=0)
        admission.reserve()

        with pytest.raises(StudioJobQueueFull) as refusal:
            admission.reserve()

        detail = refusal.value.to_public_detail()
        assert detail["error"] == "job_queue_full"
        assert detail["running"] == 1
        assert detail["limit"] == 0
        assert "queue is full" in str(detail["reason"])

    @pytest.mark.parametrize(
        ("concurrent", "queued"), [(0, 1), (-1, 1), (1, -1)], ids=["zero", "negative", "queue"]
    )
    def test_a_nonsensical_ceiling_is_refused(self, concurrent: int, queued: int) -> None:
        with pytest.raises(ValueError):
            StudioJobAdmission(max_concurrent=concurrent, max_queued=queued)


class TestManagerAdmission:
    def _manager(self, root: Path, **kwargs: object) -> StudioJobManager:
        return StudioJobManager(
            root=root,
            allowed_kinds=frozenset({"analysis"}),
            default_timeout_seconds=30.0,
            **kwargs,  # type: ignore[arg-type]
        )

    def test_overflow_is_refused_predictably_and_never_reaches_the_ledger(
        self, tmp_path: Path
    ) -> None:
        manager = self._manager(tmp_path / "jobs", max_concurrent_jobs=1, max_queued_jobs=0)
        release = threading.Event()

        def blocking(context: StudioJobContext) -> dict[str, object]:
            del context
            release.wait(30.0)
            return {"done": True}

        first = manager.submit(kind="analysis", owner="operator", request_id="req-1", task=blocking)
        try:
            with pytest.raises(StudioJobQueueFull) as refusal:
                manager.submit(
                    kind="analysis",
                    owner="operator",
                    request_id="req-2",
                    task=lambda context: {"never": True},
                )
            assert refusal.value.to_public_detail()["running"] == 1
            # A refused submission is not a job: nothing about it is recorded.
            assert [record.job_id for record in manager.list_records()] == [first.job_id]
        finally:
            release.set()

        assert manager.wait(first.job_id, timeout_seconds=30.0).status == "completed"

    def test_a_slot_is_returned_when_the_job_reaches_its_outcome(self, tmp_path: Path) -> None:
        manager = self._manager(tmp_path / "jobs", max_concurrent_jobs=1, max_queued_jobs=0)

        for index in range(3):
            record = manager.submit(
                kind="analysis",
                owner="operator",
                request_id=f"req-{index}",
                task=lambda context: {"index": 1},
            )
            assert manager.wait(record.job_id, timeout_seconds=30.0).status == "completed"

        snapshot = manager.status().to_public_dict()["admission"]
        assert snapshot["running"] == 0
        assert snapshot["admitted"] == 3
        assert snapshot["refused"] == 0

    def test_a_duplicate_submission_holds_no_slot(self, tmp_path: Path) -> None:
        manager = self._manager(tmp_path / "jobs", max_concurrent_jobs=1, max_queued_jobs=0)
        first = manager.submit(
            kind="analysis",
            owner="operator",
            request_id="req-1",
            task=lambda context: {"value": 1},
            idempotency_key="key-1",
        )
        assert manager.wait(first.job_id, timeout_seconds=30.0).status == "completed"

        again = manager.submit(
            kind="analysis",
            owner="operator",
            request_id="req-1",
            task=lambda context: {"value": 2},
            idempotency_key="key-1",
        )

        assert again.job_id == first.job_id
        # The duplicate released the slot it took, so the next job still fits.
        assert manager.status().to_public_dict()["admission"]["running"] == 0
        third = manager.submit(
            kind="analysis", owner="operator", request_id="req-2", task=lambda c: {"v": 3}
        )
        assert manager.wait(third.job_id, timeout_seconds=30.0).status == "completed"

    def test_a_rejected_kind_releases_its_slot(self, tmp_path: Path) -> None:
        from sc_neurocore.studio.platform.jobs_models import StudioJobRejected

        manager = self._manager(tmp_path / "jobs", max_concurrent_jobs=1, max_queued_jobs=0)

        with pytest.raises(StudioJobRejected):
            manager.submit(
                kind="not-allowed",
                owner="operator",
                request_id="req-1",
                task=lambda context: {},
            )

        record = manager.submit(
            kind="analysis", owner="operator", request_id="req-2", task=lambda c: {"v": 1}
        )
        assert manager.wait(record.job_id, timeout_seconds=30.0).status == "completed"


class TestResolvedCostFactors:
    def test_a_substepped_multi_state_model_costs_more_than_a_scalar_map(self) -> None:
        from sc_neurocore.studio.platform import resolve_model_cost_factors

        scalar = resolve_model_cost_factors("SCLapicqueLIFNeuron", None)
        conductance = resolve_model_cost_factors("HodgkinHuxleyNeuron", None)

        assert scalar.work_per_step == 1
        # Four gating variables and a substepped integrator: the old projection
        # budgeted this as if it were the scalar map above.
        assert conductance.state_count >= 4
        assert conductance.substeps > 1
        assert conductance.work_per_step > scalar.work_per_step

    def test_an_unresolvable_model_falls_back_instead_of_raising(self) -> None:
        from sc_neurocore.studio.platform import resolve_model_cost_factors

        factors = resolve_model_cost_factors("NoSuchNeuronExistsHere", None)

        assert factors.work_per_step == 1
        assert factors.dt > 0.0

    def test_the_requested_timestep_wins_over_the_model_default(self) -> None:
        from sc_neurocore.studio.platform import resolve_model_cost_factors

        assert resolve_model_cost_factors("HodgkinHuxleyNeuron", 0.05).dt == 0.05
