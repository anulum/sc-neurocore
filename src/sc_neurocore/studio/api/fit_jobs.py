# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Owned background fitting and experiment-cohort jobs

"""Admit laboratory tasks before submission and preserve owner-bound custody."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, cast

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict

from sc_neurocore.fitting.cohort import cohort_from_dict
from sc_neurocore.fitting.cohort_run import replay_cohort, run_cohort
from sc_neurocore.fitting.fit import fit_parameters
from sc_neurocore.fitting.problem import canonical_sha256, problem_from_dict
from sc_neurocore.studio.api.fits import (
    FitRequest,
    ReplayRequest,
    estimated_fit_steps,
    fit_problem,
    fit_replay_request,
)
from sc_neurocore.studio.api.runtime import StudioApiContext
from sc_neurocore.studio.platform.jobs_context import StudioJobContext
from sc_neurocore.studio.platform.jobs_models import StudioJobRecord, StudioJobRejected
from sc_neurocore.studio.platform.policy_models import Principal

MAX_JOB_FIT_STEPS = 200_000_000
MAX_LAB_DOCUMENT_BYTES = 4_000_000


class MeasurementRequest(BaseModel):
    """A complete scientific result and externally acquired receipts."""

    model_config = ConfigDict(extra="forbid")
    result: dict[str, Any]
    receipts: list[dict[str, Any]]


class CohortRequest(BaseModel):
    """The full versioned experiment document, without local path references."""

    model_config = ConfigDict(extra="forbid")
    cohort: dict[str, Any]


def _actor(request: Request) -> str:
    """Use authenticated identity or the explicitly supported local lab actor."""
    principal = getattr(request.state, "studio_principal", None)
    return principal.principal_id if isinstance(principal, Principal) else "local"


def _error(exc: Exception) -> HTTPException:
    return HTTPException(422, detail={"reason": "invalid_laboratory_request", "message": str(exc)})


def _bounded(document: Mapping[str, Any]) -> None:
    """Refuse oversized or nonfinite documents before worker admission."""
    if len(json.dumps(document, allow_nan=False).encode()) > MAX_LAB_DOCUMENT_BYTES:
        raise ValueError("laboratory document exceeds the submission byte budget")


def execute_laboratory_task(
    context: StudioJobContext, payload: Mapping[str, object]
) -> dict[str, object]:
    """Run an admitted scientific task with cancellation and durable artifacts.

    Parameters
    ----------
    context:
        Existing confined process-worker context.
    payload:
        Validated operation and complete exported scientific problem.

    Returns
    -------
    dict
        Replayable fit, cohort or replay result; no local filesystem paths.
    """
    operation = payload["operation"]
    document = cast(dict[str, Any], payload["document"])
    context.check_cancelled()

    def progress(event: dict[str, Any]) -> None:
        """Persist each generation/trial event and honour cancellation."""
        context.check_cancelled()
        context.append_artifact_event("history.jsonl", event)

    if operation == "fit":
        result = fit_parameters(
            problem_from_dict(document["problem"]),
            generations=int(document["generations"]),
            population=int(document["population"]),
            progress=progress,
        )
    elif operation == "cohort":
        result = run_cohort(cohort_from_dict(document), progress=progress)
    elif operation == "fit_replay":
        original = document
        result_body = fit_parameters(
            problem_from_dict(original["problem"]),
            generations=int(original["provenance"]["generations"]),
            population=int(original["provenance"]["population"]),
            progress=progress,
        )
        result = {
            "reproduced": result_body["result_sha256"] == original["result_sha256"],
            "exported_sha256": original["result_sha256"],
            "replayed_sha256": result_body["result_sha256"],
            "result": result_body,
        }
    else:
        raise ValueError("unsupported laboratory operation")
    context.check_cancelled()
    context.publish_existing_artifact("history.jsonl")
    context.write_artifact("experiment.json", json.dumps(document, allow_nan=False, sort_keys=True))
    context.write_artifact("result.json", json.dumps(result, allow_nan=False, sort_keys=True))
    return cast(dict[str, object], result)


def build_fit_jobs_router(context: StudioApiContext) -> APIRouter:
    """Register scientific jobs with the existing bounded process manager."""
    router = APIRouter()

    def submit(operation: str, document: dict[str, Any], request: Request) -> dict[str, object]:
        """Bind the submitted operation and digest to the caller's actor."""
        try:
            _bounded(document)
            record = context.studio_job_manager.submit_process_task(
                kind="analysis",
                owner=_actor(request),
                request_id=getattr(request.state, "studio_request_id", None),
                task_path="sc_neurocore.studio.api.fit_jobs:execute_laboratory_task",
                payload={"operation": operation, "document": document},
                experiment_sha256=canonical_sha256(document),
                admission={"laboratory_task": operation},
            )
        except (ValueError, StudioJobRejected) as exc:
            raise _error(exc) from exc
        return {
            "job_id": record.job_id,
            "status_route": f"/api/fits/jobs/{record.job_id}",
            "job": record.to_public_dict(),
        }

    @router.post("/api/fits/jobs", status_code=202)
    def fit_job(body: FitRequest, request: Request) -> dict[str, object]:
        """Validate the full fit and submit it without blocking the request."""
        try:
            if estimated_fit_steps(body) > MAX_JOB_FIT_STEPS:
                raise ValueError("fit exceeds the background model-step estimate budget")
            problem = fit_problem(body)
        except (KeyError, TypeError, ValueError) as exc:
            raise _error(exc) from exc
        return submit(
            "fit",
            {
                "problem": problem.to_public_dict(),
                "generations": body.generations,
                "population": body.population,
            },
            request,
        )

    @router.post("/api/fits/replay/jobs", status_code=202)
    def fit_replay_job(body: ReplayRequest, request: Request) -> dict[str, object]:
        """Admit replay under the same optimiser and sample limits as a new fit."""
        try:
            fit = fit_replay_request(body.result)
        except (KeyError, TypeError, ValueError) as exc:
            raise _error(exc) from exc
        if estimated_fit_steps(fit) > MAX_JOB_FIT_STEPS:
            raise _error(ValueError("fit replay exceeds the background model-step estimate budget"))
        return submit("fit_replay", body.result, request)

    @router.post("/api/cohorts/jobs", status_code=202)
    def cohort_job(body: CohortRequest, request: Request) -> dict[str, object]:
        """Refuse incomplete or over-budget sweeps before submitting a worker."""
        try:
            cohort = cohort_from_dict(body.cohort)
        except (KeyError, TypeError, ValueError) as exc:
            raise _error(exc) from exc
        return submit("cohort", cohort.to_public_dict(), request)

    @router.post("/api/cohorts/replay")
    def cohort_replay(body: ReplayRequest) -> dict[str, Any]:
        """Replay the full bounded cohort, including rejected and failed trials."""
        try:
            _bounded(body.result)
            return replay_cohort(body.result)
        except (KeyError, TypeError, ValueError) as exc:
            raise _error(exc) from exc

    @router.post("/api/cohorts/measurements")
    def measurements(body: MeasurementRequest) -> dict[str, Any]:
        """Compare operator-supplied measurements with explicit custody limits."""
        from sc_neurocore.fitting.pareto import measured_pareto

        try:
            _bounded(body.model_dump())
        except (TypeError, ValueError) as exc:
            raise _error(exc) from exc
        return measured_pareto(body.result, body.receipts)

    def owned(job_id: str, request: Request) -> StudioJobRecord:
        """Hide records belonging to other callers or unrelated task surfaces."""
        try:
            record = context.studio_job_manager.record(job_id)
        except (KeyError, ValueError):
            raise HTTPException(404, "laboratory job not found") from None
        if record.owner != _actor(request) or record.admission.get("laboratory_task") not in {
            "fit",
            "fit_replay",
            "cohort",
        }:
            raise HTTPException(404, "laboratory job not found")
        return record

    @router.get("/api/fits/jobs/{job_id}")
    def job_status(job_id: str, request: Request) -> dict[str, object]:
        """Read only the caller's laboratory job, including terminal failures."""
        return owned(job_id, request).to_public_dict()

    @router.post("/api/fits/jobs/{job_id}/cancel")
    def cancel_job(job_id: str, request: Request) -> dict[str, object]:
        """Cancel the caller's task through the existing process supervisor."""
        owned(job_id, request)
        return context.studio_job_manager.cancel(job_id).to_public_dict()

    return router
