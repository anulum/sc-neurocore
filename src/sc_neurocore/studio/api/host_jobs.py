# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — local hosted scientific job routes

"""Adapt closed hosted operations to the real local job service."""

from __future__ import annotations

import importlib
from typing import Annotated

from fastapi import APIRouter, Body, HTTPException, Request

from sc_neurocore.studio.api.host_job_tasks import HOST_ANALYSIS_TASK, HOST_COMPILER_TASK
from sc_neurocore.studio.api.host_jobs_schema import (
    HOST_ANALYSIS_OPERATIONS,
    HOST_COMPILER_OPERATIONS,
    HostAnalysisRequest,
    HostBridgeDescription,
    HostJobRequest,
    HostOperation,
    parse_host_job,
)
from sc_neurocore.studio.api.runtime import StudioApiContext
from sc_neurocore.studio.platform.jobs_models import StudioJobRejected
from sc_neurocore.studio.platform.policy_models import Principal


def build_host_jobs_router(context: StudioApiContext) -> APIRouter:
    """Build local-only hosted adapters under the normal middleware policies.

    Parameters
    ----------
    context : StudioApiContext
        Existing settings, policy and job-service owners; no alternate identity.

    Returns
    -------
    APIRouter
        Description, closed asynchronous submission and administrator cancellation.
    """
    router = APIRouter()
    manager = context.studio_job_manager

    def require_local_profile() -> None:
        """Refuse the hosted local-first contract in the production profile."""
        if context.settings.deployment_profile != "development":
            raise HTTPException(404, "hosted_local_profile_unavailable")

    @router.get("/api/studio/host-bridge", response_model=HostBridgeDescription)
    def describe_host_bridge() -> HostBridgeDescription:
        """Describe installed operation families without attesting scientific validity."""
        require_local_profile()
        operations: list[HostOperation] = list(HOST_ANALYSIS_OPERATIONS)
        try:
            importlib.import_module("sc_neurocore_engine.ir")
        except ImportError:
            pass
        else:
            operations.extend(HOST_COMPILER_OPERATIONS)
        return HostBridgeDescription(operations=tuple(operations))

    @router.post("/api/studio/host-jobs")
    async def submit_host_job(
        request: Request,
        _validated: Annotated[HostJobRequest, Body(discriminator="operation")],
    ) -> dict[str, object]:
        """Validate before admission and preserve the complete original JSON text."""
        require_local_profile()
        try:
            text = (await request.body()).decode("utf-8")
            body = parse_host_job(text)
        except (ValueError, RecursionError) as exc:
            raise HTTPException(422, "invalid_hosted_job_request") from exc
        analysis = isinstance(body, HostAnalysisRequest)
        try:
            record = manager.submit_process_task(
                kind="analysis" if analysis else "compiler",
                owner="studio-host-analysis" if analysis else "studio-host-compiler",
                request_id=getattr(request.state, "request_id", None),
                task_path=HOST_ANALYSIS_TASK if analysis else HOST_COMPILER_TASK,
                payload={"envelope_json": text},
            )
        except StudioJobRejected as exc:
            raise HTTPException(409, "hosted_job_admission_refused") from exc
        return record.to_public_dict()

    @router.post("/api/studio/jobs/{job_id}/cancel")
    def cancel_studio_job(job_id: str, request: Request) -> dict[str, object]:
        """Request real cancellation and return the actual settled or pending record."""
        require_local_profile()
        try:
            record = manager.record(job_id)
            if record.admission.get("laboratory_task") in {"fit", "fit_replay", "cohort"}:
                principal = getattr(request.state, "studio_principal", None)
                if not isinstance(principal, Principal) or principal.principal_id != record.owner:
                    raise KeyError(job_id)
            return manager.cancel(job_id).to_public_dict()
        except KeyError as exc:
            raise HTTPException(404, "job_not_found") from exc
        except StudioJobRejected as exc:
            raise HTTPException(409, "job_cancellation_refused") from exc

    return router
