# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Candidate job input custody and pre-admission limits

"""Check scientific request identity before durable worker allocation."""

from __future__ import annotations

import hashlib
import json

import pytest

from sc_neurocore.studio.candidate_job_contract import (
    CandidateRequest,
    CandidateSimulateRequest,
    prepare_candidate_job,
)
from sc_neurocore.studio.candidate_run import CandidateRejected, CandidateRunRefused
from tests.studio_candidate_support import adex_candidate


def test_request_bytes_preserve_order_and_are_immutable_after_admission() -> None:
    """Frozen scientific input retains the author's state/dynamics ordering."""
    document = adex_candidate()
    document["model"]["state"] = {"w": 0.0, "v": -65.0}
    document["model"]["dynamics"] = dict(reversed(tuple(document["model"]["dynamics"].items())))
    request = CandidateSimulateRequest(candidate=document, current=800.0, steps=17)
    admitted = prepare_candidate_job(request, "simulate")
    restored = json.loads(admitted.request_bytes)
    assert list(restored["candidate"]["model"]["state"]) == ["w", "v"]
    assert list(restored["candidate"]["model"]["dynamics"]) == list(document["model"]["dynamics"])
    assert admitted.request_sha256 == hashlib.sha256(admitted.request_bytes).hexdigest()
    assert admitted.projected_steps == 17
    request.candidate["model"]["parameters"]["tau_w"] = 900.0
    assert json.loads(admitted.request_bytes)["candidate"]["model"]["parameters"]["tau_w"] == 300.0


def test_operating_point_has_its_own_identity_without_relabelling_the_candidate() -> None:
    """Different currents/lengths preserve candidate identity and bind distinct jobs."""
    requests = [
        prepare_candidate_job(
            CandidateSimulateRequest(candidate=adex_candidate(), current=current, steps=steps),
            "simulate",
        )
        for current, steps in [(0.0, 10), (800.0, 10), (800.0, 20)]
    ]
    assert len({request.candidate_sha256 for request in requests}) == 1
    assert len({request.request_sha256 for request in requests}) == 3


def test_review_estimate_includes_every_proposed_reference_run() -> None:
    """Reference-job admission counts all original scientific inputs without running them."""
    request = prepare_candidate_job(CandidateRequest(candidate=adex_candidate()), "review")
    assert request.projected_steps == 6000
    assert json.loads(request.request_bytes)["candidate"] == adex_candidate()


def test_invalid_units_and_oversized_seeds_are_refused_before_submission() -> None:
    """Located dimensional faults and byte ceilings refuse the actual request contract."""
    document = adex_candidate()
    document["units"]["state"]["v"] = "ms"
    with pytest.raises(CandidateRejected) as refused:
        prepare_candidate_job(CandidateRequest(candidate=document), "review")
    assert refused.value.validation["valid"] is False
    with pytest.raises(CandidateRunRefused, match="byte budget"):
        prepare_candidate_job(
            CandidateRequest(candidate=adex_candidate()), "review", max_request_bytes=1
        )
    with pytest.raises(CandidateRunRefused, match="unsupported"):
        prepare_candidate_job(CandidateRequest(candidate=adex_candidate()), "simulate")
