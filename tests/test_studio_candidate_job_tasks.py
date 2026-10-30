# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Real candidate worker input binding refusals

"""Launch the production worker with substituted envelopes and seed bytes."""

from __future__ import annotations

from pathlib import Path

import pytest

from sc_neurocore.studio.candidate_job_contract import (
    CANDIDATE_REQUEST_SEED,
    CandidateSimulateRequest,
    prepare_candidate_job,
)
from sc_neurocore.studio.platform.jobs_manager import StudioJobManager
from tests.studio_candidate_support import adex_candidate


@pytest.mark.parametrize("mutation", ["keys", "bytes", "request", "candidate"])
def test_real_worker_refuses_substituted_scientific_inputs(mutation: str, tmp_path: Path) -> None:
    """A real process cannot publish results for bytes outside its admission."""
    admitted = prepare_candidate_job(
        CandidateSimulateRequest(candidate=adex_candidate(), steps=2), "simulate"
    )
    payload: dict[str, object] = {
        "request_sha256": admitted.request_sha256,
        "candidate_sha256": admitted.candidate_sha256,
    }
    raw = admitted.request_bytes
    if mutation == "keys":
        payload["caller_task"] = "foreign"
    elif mutation == "bytes":
        raw += b" "
    elif mutation == "request":
        payload["request_sha256"] = "0" * 64
    else:
        payload["candidate_sha256"] = "0" * 64
    manager = StudioJobManager(
        root=tmp_path / "jobs", allowed_kinds=frozenset({"analysis"}), default_timeout_seconds=30
    )
    job = manager.submit_process_task(
        kind="analysis",
        owner="alice",
        request_id="binding-refusal",
        task_path="sc_neurocore.studio.candidate_job_tasks:execute_candidate_simulation_task",
        payload=payload,
        seed_inputs={CANDIDATE_REQUEST_SEED: raw},
    )
    completed = manager.wait(job.job_id, timeout_seconds=45)
    assert completed.status == "failed", completed
    assert completed.result is None
    assert completed.artifacts == ()
    assert manager.unreaped_workers == ()
