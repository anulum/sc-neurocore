# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Actor-owned graph job routes

"""Submit, recover and control graph jobs through the existing job authority."""

from __future__ import annotations

import re

from fastapi import APIRouter, Header, HTTPException, Query, Request
from starlette.responses import Response

from sc_neurocore.studio.api.runtime import StudioApiContext
from sc_neurocore.studio.graph_job import (
    GRAPH_JOB_SCHEMA,
    GRAPH_JOB_RESULT_SCHEMA,
    GRAPH_REQUEST_SEED,
    GRAPH_RESULT_ARTIFACT,
    GraphJobRefused,
    PreparedGraphJob,
    prepare_graph_job,
)
from sc_neurocore.studio.network_graph_spec import GraphRejected
from sc_neurocore.studio.platform.jobs_admission import StudioJobQueueFull
from sc_neurocore.studio.platform.jobs_models import (
    STUDIO_JOB_ID_PATTERN,
    StudioJobConflict,
    StudioJobArtifactUnavailable,
    StudioJobRecord,
    StudioJobRejected,
)
from sc_neurocore.studio.platform.policy_models import Principal


def _actor(request: Request) -> str:
    principal = getattr(request.state, "studio_principal", None)
    return principal.principal_id if isinstance(principal, Principal) else "local"


def _graph_record(record: StudioJobRecord) -> bool:
    return record.kind == "analysis" and record.admission.get("graph_operation") == "simulate"


def build_graph_jobs_router(context: StudioApiContext) -> APIRouter:
    """Build actor-scoped graph admission, history, result and cancellation routes.

    Parameters
    ----------
    context : StudioApiContext
        Existing settings, job service and policy collaborators.

    Returns
    -------
    APIRouter
        Graph routes sharing the embedded or isolated durable job authority.
    """
    router = APIRouter()
    manager = context.studio_job_manager

    def owned(job_id: str, request: Request) -> StudioJobRecord:
        if not STUDIO_JOB_ID_PATTERN.fullmatch(job_id):
            raise HTTPException(404, "graph_job_not_found")
        try:
            record = manager.record(job_id)
        except (KeyError, PermissionError):
            raise HTTPException(404, "graph_job_not_found") from None
        except (ValueError, EOFError, OSError) as exc:
            raise HTTPException(503, "graph_job_observation_unavailable") from exc
        if record.owner != _actor(request) or not _graph_record(record):
            raise HTTPException(404, "graph_job_not_found")
        return record

    def prepare(graph: dict[str, object]) -> PreparedGraphJob:
        """Apply the same operator and service budgets without allocating job state."""
        try:
            return prepare_graph_job(
                graph,
                max_request_bytes=min(
                    context.settings.job_max_artifact_bytes, manager.max_seed_file_bytes
                ),
            )
        except GraphRejected as exc:
            raise HTTPException(422, detail=exc.to_public_detail()) from exc
        except GraphJobRefused as exc:
            raise HTTPException(
                422, detail={"error": "graph_job_refused", "reason": str(exc)}
            ) from exc

    @router.post("/api/graph/jobs/validate")
    def validate_graph_job(graph: dict[str, object]) -> dict[str, object]:
        """Resolve asynchronous identity and seal exact input without reserving or running work."""
        prepared = prepare(graph)
        return {
            "schema_version": "studio.graph.job.validation.v1",
            "execution_mode": "async_job",
            "graph_sha256": prepared.graph_sha256,
            "request_sha256": prepared.request_sha256,
            "submission_sha256": prepared.submission_sha256,
            "request_seal_sha256": prepared.request_seal_sha256,
            "projected_neuron_steps": prepared.spec.neuron_steps,
            "n_steps": prepared.spec.n_steps,
            "n_neurons": prepared.spec.n_neurons,
        }

    @router.post("/api/graph/jobs")
    def submit_graph_job(
        graph: dict[str, object],
        request: Request,
        idempotency_key: str | None = Header(
            default=None, min_length=1, max_length=128, pattern=r"^[!-~]+$"
        ),
        expected_submission_sha256: str | None = Header(
            default=None,
            alias="X-Studio-Graph-Submission-SHA256",
            pattern=r"^[0-9a-f]{64}$",
        ),
    ) -> dict[str, object]:
        """Admit an exact graph before allocation and bind retries to its specification."""
        prepared = prepare(graph)
        if (
            expected_submission_sha256 is not None
            and expected_submission_sha256 != prepared.submission_sha256
        ):
            raise HTTPException(409, "graph_job_preparation_changed")
        try:
            record = manager.submit_process_task(
                kind="analysis",
                owner=_actor(request),
                request_id=getattr(request.state, "studio_request_id", None),
                task_path="sc_neurocore.studio.graph_job:run_graph_job",
                payload={
                    "graph_sha256": prepared.graph_sha256,
                    "request_sha256": prepared.request_sha256,
                },
                seed_inputs={GRAPH_REQUEST_SEED: prepared.request_bytes},
                idempotency_key=idempotency_key,
                experiment_sha256=prepared.submission_sha256,
                admission={
                    "graph_operation": "simulate",
                    "graph_sha256": prepared.graph_sha256,
                    "request_sha256": prepared.request_sha256,
                    "projected_neuron_steps": prepared.spec.neuron_steps,
                },
            )
        except StudioJobConflict as exc:
            raise HTTPException(409, "graph_job_idempotency_conflict") from exc
        except StudioJobQueueFull as exc:
            raise HTTPException(503, detail=exc.to_public_detail()) from exc
        except PermissionError as exc:
            raise HTTPException(403, "graph_job_admission_refused") from exc
        except StudioJobRejected as exc:
            raise HTTPException(503, "graph_job_admission_failed") from exc
        except (KeyError, ValueError, EOFError, OSError) as exc:
            raise HTTPException(503, "graph_job_admission_unconfirmed") from exc
        if record.experiment_sha256 != prepared.submission_sha256 or not _graph_record(record):
            raise HTTPException(409, "graph_job_idempotency_conflict")
        return {
            "schema_version": GRAPH_JOB_SCHEMA,
            "execution_mode": "async_job",
            "graph_sha256": prepared.graph_sha256,
            "request_sha256": prepared.request_sha256,
            "projected_neuron_steps": prepared.spec.neuron_steps,
            "job_id": record.job_id,
            "job": record.to_public_dict(),
            "status_route": f"/api/graph/jobs/{record.job_id}",
        }

    @router.get("/api/graph/jobs")
    def graph_job_history(
        request: Request, limit: int = Query(default=50, ge=1, le=100)
    ) -> dict[str, object]:
        """List the requesting actor's recent graph jobs without large result arrays."""
        records = tuple(
            record
            for record in manager.list_records(actor=_actor(request))
            if _graph_record(record)
        )
        return {
            "schema_version": "studio.graph.jobs.v1",
            "jobs": [
                {**record.to_public_dict(), "result": None} for record in reversed(records[-limit:])
            ],
        }

    @router.get("/api/graph/jobs/{job_id}")
    def graph_job_status(job_id: str, request: Request) -> dict[str, object]:
        """Read durable graph state and its compact complete-result receipt after restart."""
        return owned(job_id, request).to_public_dict()

    @router.post("/api/graph/jobs/{job_id}/cancel")
    def graph_job_cancel(job_id: str, request: Request) -> dict[str, object]:
        """Request supervisor cancellation without fabricating an immediate terminal state."""
        owned(job_id, request)
        try:
            return manager.cancel(job_id).to_public_dict()
        except (KeyError, PermissionError):
            raise HTTPException(404, "graph_job_not_found") from None
        except StudioJobRejected as exc:
            raise HTTPException(403, "graph_job_cancellation_refused") from exc
        except (ValueError, EOFError, OSError) as exc:
            raise HTTPException(503, "graph_job_cancellation_unconfirmed") from exc

    @router.get("/api/graph/jobs/{job_id}/result")
    def graph_job_result(job_id: str, request: Request) -> Response:
        """Download the actor's complete declared graph JSON after integrity verification."""
        record = owned(job_id, request)
        if record.status != "completed":
            raise HTTPException(409, "graph_job_result_unavailable")
        try:
            artifact = manager.read_artifact(job_id, GRAPH_RESULT_ARTIFACT)
        except (KeyError, PermissionError):
            raise HTTPException(404, "graph_job_result_not_found") from None
        except (StudioJobArtifactUnavailable, StudioJobRejected, ValueError):
            raise HTTPException(409, "graph_job_result_unavailable") from None
        except (EOFError, OSError) as exc:
            raise HTTPException(503, "graph_job_result_unconfirmed") from exc
        # The service has verified the bytes against its sealed manifest; bind
        # that same artifact to the scientific receipt on every download.
        receipt = record.result
        expected_graph = record.admission.get("graph_sha256")
        if (
            not isinstance(receipt, dict)
            or receipt.get("schema_version") != GRAPH_JOB_RESULT_SCHEMA
            or not isinstance(expected_graph, str)
            or re.fullmatch(r"[0-9a-f]{64}", expected_graph) is None
            or receipt.get("graph_sha256") != expected_graph
            or receipt.get("result_artifact") != GRAPH_RESULT_ARTIFACT
            or receipt.get("result_sha256") != artifact.artifact.sha256
            or type(receipt.get("result_size_bytes")) is not int
            or receipt.get("result_size_bytes") != artifact.artifact.size_bytes
        ):
            raise HTTPException(409, "graph_job_result_unavailable")
        return Response(
            artifact.payload,
            media_type="application/json",
            headers={
                "Content-Disposition": 'attachment; filename="network-result.json"',
                "X-Studio-Artifact-SHA256": artifact.artifact.sha256,
                "X-Studio-Artifact-Size": str(artifact.artifact.size_bytes),
            },
        )

    return router
