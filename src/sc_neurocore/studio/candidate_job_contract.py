# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Immutable candidate job request admission

"""Validate candidate work before admission and preserve its ordered inputs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

from sc_neurocore.studio.candidate_package import MAX_CANDIDATE_STEPS, candidate_sha256
from sc_neurocore.studio.candidate_run import CandidateRunRefused, require_valid_candidate

CandidateOperation = Literal["simulate", "review"]
MAX_CANDIDATE_REQUEST_BYTES = 4_000_000
CANDIDATE_REQUEST_SEED = "candidate-request.json"
CANDIDATE_JOB_SCHEMA = "studio.candidate.job.v1"


class CandidateRequest(BaseModel):
    """A complete candidate package, without caller-selected task authority."""

    model_config = ConfigDict(extra="forbid")
    candidate: dict[str, Any]


class CandidateSimulateRequest(CandidateRequest):
    """A candidate and its finite constant current and bounded step count."""

    current: FiniteFloat = Field(default=0.0, strict=True)
    steps: int = Field(default=1000, ge=1, le=MAX_CANDIDATE_STEPS)


@dataclass(frozen=True, slots=True)
class CandidateJobInput:
    """Validated immutable request bytes and their independent custody digests."""

    operation: CandidateOperation
    request_bytes: bytes
    request_sha256: str
    candidate_sha256: str
    candidate_name: str
    projected_steps: int


def prepare_candidate_job(
    request: CandidateRequest,
    operation: CandidateOperation,
    *,
    max_request_bytes: int = MAX_CANDIDATE_REQUEST_BYTES,
) -> CandidateJobInput:
    """Validate a candidate and freeze the exact request before allocating a job.

    Parameters
    ----------
    request : CandidateRequest
        Complete candidate; simulation additionally carries current and steps.
    operation : {"simulate", "review"}
        Selected registered operation, never an import path from the caller.
    max_request_bytes : int
        Submission ceiling, further restricted by the configured artifact limit.

    Returns
    -------
    CandidateJobInput
        Ordered UTF-8 JSON, request and candidate hashes, and model-step estimate.

    Raises
    ------
    CandidateRejected
        Candidate admission fails with located diagnostics.
    CandidateRunRefused
        Operation, request shape or encoded byte budget is invalid.
    """
    if operation not in {"simulate", "review"} or (
        operation == "simulate" and not isinstance(request, CandidateSimulateRequest)
    ):
        raise CandidateRunRefused("unsupported candidate job request")
    require_valid_candidate(request.candidate)
    encoded = json.dumps(
        request.model_dump(), ensure_ascii=False, allow_nan=False, separators=(",", ":")
    ).encode("utf-8")
    if len(encoded) > min(MAX_CANDIDATE_REQUEST_BYTES, max_request_bytes):
        raise CandidateRunRefused("candidate request exceeds the submission byte budget")
    projected = (
        request.steps
        if isinstance(request, CandidateSimulateRequest) and operation == "simulate"
        else sum(int(test["steps"]) for test in request.candidate["reference_tests"])
    )
    return CandidateJobInput(
        operation=operation,
        request_bytes=encoded,
        request_sha256=hashlib.sha256(encoded).hexdigest(),
        candidate_sha256=candidate_sha256(request.candidate),
        candidate_name=str(request.candidate["name"]),
        projected_steps=projected,
    )
