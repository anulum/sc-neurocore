# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio catalogue and benchmark routes

"""Expose catalogue, template, DCLS, and benchmark-databank adapters."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from sc_neurocore.studio.api.analysis_guards import _guard_model_scan_request
from sc_neurocore.studio.api.common import _safe
from sc_neurocore.studio.api.runtime import StudioApiContext
from sc_neurocore.studio.api.schemas import (
    BenchmarkContributeRequest,
    BenchmarkRunRequest,
    DclsEvaluateRequest,
)
from sc_neurocore.studio.platform.jobs_context import StudioJobContext
from sc_neurocore.studio.platform.jobs_models import StudioJobRejected
from sc_neurocore.studio.benchmark_contribution import (
    ALLOWED_ENVIRONMENT_KEYS,
    FORBIDDEN_KEYS,
    SUBMISSION_SCHEMA_VERSION,
    databank_leaderboard,
    run_local_benchmark,
    store_contribution,
)
from sc_neurocore.studio.dcls import (
    dcls_benchmark,
    dcls_forward_parity,
    dcls_kernel_info,
    dcls_tent_profile,
)
from sc_neurocore.studio.model_scan import scan_all_models
from sc_neurocore.studio.models import (
    get_model_detail,
    list_models,
    model_documentation,
    model_facets,
)
from sc_neurocore.studio.templates import get_template, list_templates


def build_catalogue_router(context: StudioApiContext) -> APIRouter:
    """Build the catalogue and benchmark router over shared Studio runtime state."""
    router = APIRouter()
    analysis_budget = context.analysis_budget
    studio_job_manager = context.studio_job_manager

    @router.get("/api/templates")
    def api_templates() -> list[dict[str, Any]]:
        return list_templates()

    @router.get("/api/templates/{name}")
    def api_template(name: str) -> Any:
        t = get_template(name)
        if not t:
            raise HTTPException(404, f"Template '{name}' not found")
        return t

    @router.get("/api/models")
    def api_models() -> Any:
        return _safe(list_models)

    @router.get("/api/models/scan")
    def api_model_scan() -> Any:
        """Run a budgeted synchronous catalogue model scan.

        Full-catalogue scans that exceed the synchronous analysis budget are
        rejected with ``execution_mode=job_required`` so operators use
        :func:`api_model_scan_job` instead of blocking the HTTP request thread.
        """

        duration = 100.0
        _guard_model_scan_request(
            analysis_budget,
            model_count=len(list_models()),
            duration=duration,
        )
        return _safe(lambda: scan_all_models(current=10.0, duration=duration))

    @router.post("/api/models/scan/jobs")
    def api_model_scan_job() -> dict[str, Any]:
        """Submit a full-catalogue model scan as an asynchronous Studio job.

        Returns a path-free job record immediately. Poll
        ``GET /api/studio/jobs/{job_id}`` for completion; the completed job
        ``result`` carries ``studio.model-scan.v1`` payload with evidence class.
        """

        duration = 100.0
        current = 10.0

        def _task(_job_context: StudioJobContext) -> dict[str, object]:
            payload = scan_all_models(current=current, duration=duration)
            return dict(payload)

        try:
            record = studio_job_manager.submit(
                kind="model_scan",
                owner="studio",
                request_id=None,
                task=_task,
            )
        except StudioJobRejected as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from None
        public = record.to_public_dict()
        return {
            "execution_mode": "async_job",
            "job": public,
            "job_id": record.job_id,
            "schema_version": "studio.model-scan.job.v1",
            "status_route": f"/api/studio/jobs/{record.job_id}",
        }

    @router.get("/api/models/facets")
    def api_model_facets() -> Any:
        return _safe(model_facets)

    @router.get("/api/models/{name}/doc")
    def api_model_doc(name: str) -> Any:
        return _safe(
            lambda: (
                model_documentation(name)
                or (_ for _ in ()).throw(HTTPException(404, f"No documentation for model '{name}'"))
            )
        )

    @router.get("/api/dcls/info")
    def api_dcls_info() -> Any:
        return _safe(dcls_kernel_info)

    @router.get("/api/dcls/benchmark")
    def api_dcls_benchmark() -> Any:
        return _safe(
            lambda: (
                dcls_benchmark()
                or (_ for _ in ()).throw(HTTPException(404, "No recorded DCLS benchmark available"))
            )
        )

    @router.get("/api/benchmarks/schema")
    def api_benchmark_schema() -> Any:
        return {
            "schema_version": SUBMISSION_SCHEMA_VERSION,
            "allowed_environment_keys": sorted(ALLOWED_ENVIRONMENT_KEYS),
            "forbidden_keys": sorted(FORBIDDEN_KEYS),
            "consent": "opt-in; nothing is submitted unless you choose to contribute",
        }

    @router.post("/api/benchmarks/run")
    def api_benchmark_run(run_request: BenchmarkRunRequest) -> Any:
        return _safe(
            lambda: run_local_benchmark(
                n_channels=run_request.n_channels,
                n_taps=run_request.n_taps,
                repeats=run_request.repeats,
            )
        )

    @router.post("/api/benchmarks/contribute")
    def api_benchmark_contribute(contribute_request: BenchmarkContributeRequest) -> Any:
        def _contribute() -> dict[str, Any]:
            try:
                return store_contribution(contribute_request.submission, contribute_request.handle)
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from exc

        return _safe(_contribute)

    @router.get("/api/benchmarks/databank")
    def api_benchmark_databank() -> Any:
        return _safe(databank_leaderboard)

    @router.post("/api/dcls/evaluate")
    def api_dcls_evaluate(dcls_request: DclsEvaluateRequest) -> Any:
        def _evaluate() -> dict[str, Any]:
            profile = dcls_tent_profile(
                dcls_request.centre_q88, dcls_request.sigma_q88, dcls_request.n_taps
            )
            spikes = (
                dcls_request.spikes
                if dcls_request.spikes is not None
                else [1] * dcls_request.n_taps
            )
            weights = (
                dcls_request.weights_q88
                if dcls_request.weights_q88 is not None
                else [256] * len(spikes)
            )
            if len(spikes) != len(weights):
                raise HTTPException(400, "spikes and weights_q88 must have equal length")
            forward = dcls_forward_parity(
                spikes, weights, dcls_request.centre_q88, dcls_request.sigma_q88
            )
            return {"profile": profile, "forward": forward}

        return _safe(_evaluate)

    @router.get("/api/models/{name}")
    def api_model(name: str) -> Any:
        return _safe(
            lambda: (
                get_model_detail(name)
                or (_ for _ in ()).throw(HTTPException(404, f"Model '{name}' not found"))
            )
        )

    return router
