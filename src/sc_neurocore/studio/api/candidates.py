# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio routes for candidate model packages

"""Validate, diff, simulate and review candidate model packages.

The routes are stateless: a candidate travels in the request and is kept by
the workspace that holds it. None of them writes a canonical file or changes
the catalogue. An invalid candidate is refused by the three routes that act on
it with HTTP 422 and the located diagnostics, so the editor can mark each
field; the validation route reports the diagnostics with HTTP 200.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from sc_neurocore.studio.api.runtime import StudioApiContext
from sc_neurocore.studio.api.schemas import FiniteFloat
from sc_neurocore.studio.candidate_package import MAX_CANDIDATE_STEPS


class CandidateRequest(BaseModel):
    """A candidate package, sent whole."""

    model_config = ConfigDict(extra="forbid")

    candidate: dict[str, Any]


class CandidateSimulateRequest(CandidateRequest):
    """A candidate package and the run to perform with it."""

    current: FiniteFloat = 0.0
    steps: int = Field(default=1000, ge=1, le=MAX_CANDIDATE_STEPS)


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
        Shared runtime state; the candidate routes hold none of their own.
    """
    del context
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

    @router.post("/api/candidates/simulate")
    def api_candidate_simulate(request: CandidateSimulateRequest) -> dict[str, Any]:
        """Simulate a valid candidate under its own profile and a constant current."""
        from sc_neurocore.studio.candidate_run import simulate_candidate

        return _acting_on_valid(
            lambda: simulate_candidate(
                request.candidate, current=request.current, steps=request.steps
            )
        )

    @router.post("/api/candidates/review-packet")
    def api_candidate_review_packet(request: CandidateRequest) -> dict[str, Any]:
        """Run a valid candidate's reference tests and return its review packet."""
        from sc_neurocore.studio.candidate_run import review_packet

        return _acting_on_valid(lambda: review_packet(request.candidate))

    return router


__all__ = ["CandidateRequest", "CandidateSimulateRequest", "build_candidates_router"]
