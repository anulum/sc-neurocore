# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Candidate simulation and review process tasks

"""Execute admitted candidate requests using the unchanged scientific runtime."""

from __future__ import annotations

import json
from collections.abc import Mapping

from sc_neurocore.studio.candidate_job_contract import (
    CANDIDATE_REQUEST_SEED,
    CandidateOperation,
    CandidateRequest,
    CandidateSimulateRequest,
    prepare_candidate_job,
)
from sc_neurocore.studio.candidate_run import CandidateRunRefused, review_packet, simulate_candidate
from sc_neurocore.studio.platform.jobs_context import StudioJobContext


def _execute(
    context: StudioJobContext, payload: Mapping[str, object], operation: CandidateOperation
) -> dict[str, object]:
    context.check_cancelled()
    if set(payload) != {"request_sha256", "candidate_sha256"}:
        raise CandidateRunRefused("invalid candidate job envelope")
    raw = context.read_seed_input(CANDIDATE_REQUEST_SEED)
    request: CandidateRequest = (
        CandidateSimulateRequest.model_validate_json(raw)
        if operation == "simulate"
        else CandidateRequest.model_validate_json(raw)
    )
    admitted = prepare_candidate_job(request, operation)
    if (
        raw != admitted.request_bytes
        or admitted.request_sha256 != payload["request_sha256"]
        or admitted.candidate_sha256 != payload["candidate_sha256"]
    ):
        raise CandidateRunRefused("candidate job inputs do not match their admitted digests")
    context.write_artifact("request.json", raw)
    context.write_artifact(
        "candidate.json",
        json.dumps(request.candidate, ensure_ascii=False, allow_nan=False, separators=(",", ":")),
    )
    context.check_cancelled()
    result = (
        simulate_candidate(request.candidate, current=request.current, steps=request.steps)
        if isinstance(request, CandidateSimulateRequest)
        else review_packet(request.candidate)
    )
    context.check_cancelled()
    context.write_artifact("result.json", json.dumps(result, ensure_ascii=False, allow_nan=False))
    return dict(result)


def execute_candidate_simulation_task(
    context: StudioJobContext, payload: Mapping[str, object]
) -> dict[str, object]:
    """Run one digest-bound candidate simulation in its supervised worker.

    Parameters
    ----------
    context : StudioJobContext
        Existing bounded process context and confined input/artifact custody.
    payload : mapping
        Exactly the admitted request and candidate SHA-256 digests.

    Returns
    -------
    dict[str, object]
        Unchanged simulation result; request, candidate and result are artifacts.

    Raises
    ------
    ValueError
        The seed, candidate, envelope or digest binding is invalid.
    StudioJobCancelled
        Cancellation was observed before or after scientific execution.
    """
    return _execute(context, payload, "simulate")


def execute_candidate_review_task(
    context: StudioJobContext, payload: Mapping[str, object]
) -> dict[str, object]:
    """Run every proposed reference test and preserve the original review digest.

    Parameters
    ----------
    context : StudioJobContext
        Existing supervised process context with confined artifact custody.
    payload : mapping
        Exactly the admitted request and candidate SHA-256 digests.

    Returns
    -------
    dict[str, object]
        Existing review packet, without catalogue promotion or independent grade.

    Raises
    ------
    ValueError
        The request, candidate or digest binding is invalid.
    StudioJobCancelled
        Cancellation was observed before or after scientific execution.
    """
    return _execute(context, payload, "review")
