# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio routes for candidate model packages

"""Validate, diff, simulate and review candidate model packages.

Candidates travel in complete requests and remain workspace-local proposals.
Validation and diff do not execute the model. Simulation and reference review
return HTTP 202 after admission into the existing durable process-job ledger.
No route modifies the catalogue. Invalid candidates retain their located
HTTP 422 diagnostics; validation reports those diagnostics with HTTP 200.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from sc_neurocore.studio.api.runtime import StudioApiContext
from sc_neurocore.studio.api.candidate_jobs import submit_candidate_job
from sc_neurocore.studio.candidate_job_contract import CandidateRequest, CandidateSimulateRequest


def _acting_on_valid(action: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    """Run ``action``, turning a refused candidate into HTTP 422."""
    from sc_neurocore.studio.candidate_run import CandidateRejected

    try:
        return action()
    except CandidateRejected as rejected:
        raise HTTPException(
            status_code=422,
            detail={"reason": "invalid_candidate", "validation": rejected.validation},
        ) from rejected


def build_candidates_router(context: StudioApiContext) -> APIRouter:
    """Build the candidate-package router.

    Parameters
    ----------
    context:
        Shared runtime, policy and durable job custody.
    """
    router = APIRouter()

    @router.post("/api/candidates/validate")
    def api_candidate_validate(request: CandidateRequest) -> dict[str, Any]:
        """Report every problem with a candidate, each located by a JSON pointer."""
        from sc_neurocore.studio.candidate_package import validate_candidate

        return validate_candidate(request.candidate).to_public_dict()

    @router.post("/api/candidates/diff")
    def api_candidate_diff(request: CandidateRequest) -> dict[str, Any]:
        """Diff a valid candidate against its parent, mathematically and semantically."""
        from sc_neurocore.studio.candidate_diff import diff_candidate
        from sc_neurocore.studio.candidate_run import require_valid_candidate

        def act() -> dict[str, Any]:
            require_valid_candidate(request.candidate)
            return diff_candidate(request.candidate)

        return _acting_on_valid(act)

    @router.post("/api/candidates/simulate", status_code=202)
    def api_candidate_simulate(
        body: CandidateSimulateRequest, request: Request
    ) -> dict[str, object]:
        """Admit a valid simulation as an owned, bounded durable process job."""
        return submit_candidate_job(context, body, "simulate", request)

    @router.post("/api/candidates/review-packet", status_code=202)
    def api_candidate_review_packet(body: CandidateRequest, request: Request) -> dict[str, object]:
        """Admit the complete reference review as an owned durable process job."""
        return submit_candidate_job(context, body, "review", request)

    return router


__all__ = ["CandidateRequest", "CandidateSimulateRequest", "build_candidates_router"]
