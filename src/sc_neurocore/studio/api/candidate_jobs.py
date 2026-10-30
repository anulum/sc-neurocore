# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Requester-owned candidate job admission and recovery routes

"""Submit candidate workers and expose only the requesting actor's custody."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request
from starlette.responses import Response

from sc_neurocore.studio.api.runtime import StudioApiContext
from sc_neurocore.studio.candidate_job_contract import (
    CANDIDATE_JOB_SCHEMA,
    CANDIDATE_REQUEST_SEED,
    CandidateOperation,
    CandidateRequest,
    prepare_candidate_job,
)
from sc_neurocore.studio.candidate_run import CandidateRejected, CandidateRunRefused
from sc_neurocore.studio.platform.jobs_models import (
    STUDIO_JOB_ID_PATTERN,
    StudioJobArtifactUnavailable,
    StudioJobRecord,
    StudioJobRejected,
)
from sc_neurocore.studio.platform.policy_models import Principal


def _actor(request: Request) -> str:
    principal = getattr(request.state, "studio_principal", None)
    return principal.principal_id if isinstance(principal, Principal) else "local"


def _candidate_record(record: StudioJobRecord) -> bool:
    return record.kind == "analysis" and record.admission.get("candidate_operation") in (
        "simulate",
        "review",
    )


def submit_candidate_job(
    context: StudioApiContext,
    body: CandidateRequest,
    operation: CandidateOperation,
    request: Request,
) -> dict[str, object]:
    """Admit a complete ordered candidate request before starting a process.

    Parameters
    ----------
    context : StudioApiContext
        Existing job, settings and policy collaborators.
    body : CandidateRequest
        Complete scientific request, validated before capacity allocation.
    operation : {"simulate", "review"}
        Registered candidate task selected by the route.
    request : Request
        Policy-authenticated principal and middleware request trace.

    Returns
    -------
    dict[str, object]
        Versioned receipt, input digests, projected model steps and status route.

    Raises
    ------
    HTTPException
        Located 422 for an invalid candidate, 422 for byte admission, or 503
        when the existing manager refuses capacity or process admission.
    """
    try:
        admitted = prepare_candidate_job(
            body, operation, max_request_bytes=context.settings.job_max_artifact_bytes
        )
    except CandidateRejected as exc:
        raise HTTPException(
            422, detail={"reason": "invalid_candidate", "validation": exc.validation}
        ) from exc
    except CandidateRunRefused as exc:
        raise HTTPException(
            422, detail={"reason": "invalid_candidate_job", "message": str(exc)}
        ) from exc
    task = (
        "execute_candidate_simulation_task"
        if operation == "simulate"
        else "execute_candidate_review_task"
    )
    try:
        record = context.studio_job_manager.submit_process_task(
            kind="analysis",
            owner=_actor(request),
            request_id=getattr(request.state, "studio_request_id", None),
            task_path=f"sc_neurocore.studio.candidate_job_tasks:{task}",
            payload={
                "request_sha256": admitted.request_sha256,
                "candidate_sha256": admitted.candidate_sha256,
            },
            seed_inputs={CANDIDATE_REQUEST_SEED: admitted.request_bytes},
            experiment_sha256=admitted.request_sha256,
            admission={
                "candidate_operation": operation,
                "candidate_name": admitted.candidate_name,
                "candidate_sha256": admitted.candidate_sha256,
                "request_sha256": admitted.request_sha256,
                "projected_steps": admitted.projected_steps,
            },
        )
    except StudioJobRejected as exc:
        raise HTTPException(
            503,
            detail={
                "reason": "candidate_job_rejected",
                "message": "Candidate job admission failed.",
            },
        ) from exc
    return {
        "schema_version": CANDIDATE_JOB_SCHEMA,
        "operation": operation,
        "candidate_sha256": admitted.candidate_sha256,
        "request_sha256": admitted.request_sha256,
        "projected_steps": admitted.projected_steps,
        "job_id": record.job_id,
        "job": record.to_public_dict(),
        "status_route": f"/api/candidates/jobs/{record.job_id}",
    }


def build_candidate_jobs_router(context: StudioApiContext) -> APIRouter:
    """Build actor-scoped candidate history, observation, control and downloads.

    Parameters
    ----------
    context : StudioApiContext
        Shared durable job service; this router owns no alternate ledger.

    Returns
    -------
    APIRouter
        Registered candidate custody routes for embedded and isolated profiles.
    """
    router = APIRouter()
    manager = context.studio_job_manager

    def owned(job_id: str, request: Request) -> StudioJobRecord:
        if not STUDIO_JOB_ID_PATTERN.fullmatch(job_id):
            raise HTTPException(404, "candidate_job_not_found")
        try:
            record = manager.record(job_id)
        except (KeyError, ValueError):
            raise HTTPException(404, "candidate_job_not_found") from None
        if record.owner != _actor(request) or not _candidate_record(record):
            raise HTTPException(404, "candidate_job_not_found")
        return record

    @router.get("/api/candidates/jobs")
    def candidate_job_history(
        request: Request, limit: int = Query(default=50, ge=1, le=100)
    ) -> dict[str, object]:
        """List the actor's most recent candidate jobs without full result arrays."""
        records = tuple(
            record
            for record in manager.list_records(actor=_actor(request))
            if _candidate_record(record)
        )
        return {
            "schema_version": "studio.candidate.jobs.v1",
            "jobs": [
                {**record.to_public_dict(), "result": None} for record in reversed(records[-limit:])
            ],
        }

    @router.get("/api/candidates/jobs/{job_id}")
    def candidate_job_status(job_id: str, request: Request) -> dict[str, object]:
        """Recover the actor's actual state and stored result, including interruptions."""
        return owned(job_id, request).to_public_dict()

    @router.post("/api/candidates/jobs/{job_id}/cancel")
    def candidate_job_cancel(job_id: str, request: Request) -> dict[str, object]:
        """Request supervisor cancellation; retain the actual returned job state."""
        owned(job_id, request)
        return manager.cancel(job_id).to_public_dict()

    @router.get("/api/candidates/jobs/{job_id}/artifacts/{artifact_path:path}")
    def candidate_job_artifact(job_id: str, artifact_path: str, request: Request) -> Response:
        """Download one actor-owned declared artifact after verifying its digest."""
        owned(job_id, request)
        try:
            artifact = manager.read_artifact(job_id, artifact_path)
        except KeyError:
            raise HTTPException(404, "candidate_job_artifact_not_found") from None
        except (StudioJobArtifactUnavailable, ValueError):
            raise HTTPException(409, "candidate_job_artifact_unavailable") from None
        return Response(
            artifact.payload,
            media_type="application/json",
            headers={
                "Content-Disposition": f'attachment; filename="{Path(artifact.artifact.relative_path).name}"',
                "X-Studio-Artifact-SHA256": artifact.artifact.sha256,
                "X-Studio-Artifact-Size": str(artifact.artifact.size_bytes),
            },
        )

    return router
