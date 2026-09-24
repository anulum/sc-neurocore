# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Asynchronous Studio analysis job validation and execution

"""Validate and execute heavy analysis jobs off the HTTP request thread."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import ValidationError

from sc_neurocore.studio.analysis import (
    bifurcation_sweep,
    fi_curve_sweep,
    heatmap_2d,
    sensitivity_analysis,
)
from sc_neurocore.studio.api.analysis_guards import (
    _attach_analysis_metadata,
    _make_simulate_fn,
)
from sc_neurocore.studio.api.schemas import (
    AnalysisJobRequest,
    BifurcationRequest,
    FICurveRequest,
    HeatmapRequest,
    ModelSimulateRequest,
    SensitivityRequest,
    SimulateRequest,
)
from sc_neurocore.studio.experiment_spec import (
    JOB_MAX_STEPS,
    ExperimentRejected,
    resolve_experiment,
    run_experiment,
)
from sc_neurocore.studio.model_run_contract import ModelInputError
from sc_neurocore.studio.evidence_receipt import attach_evidence_receipt
from sc_neurocore.studio.evidence_scope import simulation_scope
from sc_neurocore.studio.simulation_manifest import build_simulation_run_manifest
from sc_neurocore.studio.platform.jobs_context import StudioJobContext
from sc_neurocore.studio.platform.jobs_models import StudioJobRejected
from sc_neurocore.studio.platform.studio_job_service import StudioJobService

AnalysisKind = Literal["fi_curve", "bifurcation", "heatmap", "sensitivity", "simulate"]


class AnalysisJobValidationError(ValueError):
    """Raised when an analysis job request payload is invalid."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code

    def to_public_detail(self) -> dict[str, str]:
        """Return a path-free public error detail."""
        return {"error": self.code}


def validate_analysis_job_request(
    req: AnalysisJobRequest,
) -> tuple[AnalysisKind, dict[str, Any], int, float, float]:
    """Validate the job request and return kind, payload dump, and cost hints.

    Returns
    -------
    tuple
        ``(analysis, payload_dump, projected_simulations, duration_ms, dt_ms)``.

    Raises
    ------
    AnalysisJobValidationError
        When the payload does not match the selected analysis schema.
    """
    analysis = req.analysis
    try:
        if analysis == "simulate":
            body: SimulateRequest | ModelSimulateRequest = (
                SimulateRequest.model_validate(req.payload)
                if "equations" in req.payload
                else ModelSimulateRequest.model_validate(req.payload)
            )
            try:
                spec = resolve_experiment(body.model_dump(), max_steps=JOB_MAX_STEPS)
            except (ExperimentRejected, ModelInputError) as exc:
                raise AnalysisJobValidationError("invalid_analysis_payload") from exc
            return analysis, body.model_dump(), 1, spec.duration_ms, spec.dt
        if analysis == "fi_curve":
            fi_body = FICurveRequest.model_validate(req.payload)
            return analysis, fi_body.model_dump(), fi_body.i_steps, fi_body.duration, fi_body.dt
        if analysis == "bifurcation":
            bif_body = BifurcationRequest.model_validate(req.payload)
            return (
                analysis,
                bif_body.model_dump(),
                bif_body.sweep_steps,
                bif_body.duration,
                bif_body.dt,
            )
        if analysis == "heatmap":
            heat_body = HeatmapRequest.model_validate(req.payload)
            return (
                analysis,
                heat_body.model_dump(),
                heat_body.x_steps * heat_body.y_steps,
                heat_body.duration,
                heat_body.dt,
            )
        sens_body = SensitivityRequest.model_validate(req.payload)
        return (
            analysis,
            sens_body.model_dump(),
            1 + 2 * len(sens_body.params or {}),
            sens_body.duration,
            sens_body.dt,
        )
    except ValidationError as exc:
        raise AnalysisJobValidationError("invalid_analysis_payload") from exc


def run_analysis_job_task(
    analysis: AnalysisKind,
    payload_dump: dict[str, Any],
    _job_context: StudioJobContext,
) -> dict[str, object]:
    """Execute one validated analysis payload and return a public result dict."""
    if analysis == "simulate":
        spec = resolve_experiment(payload_dump, max_steps=JOB_MAX_STEPS)
        result = run_experiment(spec)
        result["cache"] = {"hit": False, "key": spec.experiment_sha256}
        result["run_metadata"] = build_simulation_run_manifest(
            source=spec.source,
            request_payload=payload_dump,
            result_payload=result,
        ).to_public_dict()
        return attach_evidence_receipt(
            result,
            lane="simulation",
            status="completed",
            binding="produced",
            scope=simulation_scope(result),
        )
    if analysis == "fi_curve":
        fi = FICurveRequest.model_validate(payload_dump)
        sim_fn = _make_simulate_fn(fi.model_dump())
        result = _attach_analysis_metadata(
            "fi_curve", fi.model_dump(), fi_curve_sweep(sim_fn, fi.i_min, fi.i_max, fi.i_steps)
        )
        return dict(result)
    if analysis == "bifurcation":
        bif = BifurcationRequest.model_validate(payload_dump)
        sim_fn = _make_simulate_fn(bif.model_dump())
        base_cfg = {
            "params": bif.params,
            "init": bif.init,
            "dt": bif.dt,
            "duration": bif.duration,
            "current": bif.current,
            "protocol": "sine",
        }
        sweep = bifurcation_sweep(
            sim_fn,
            base_cfg,
            bif.sweep_param,
            bif.sweep_min,
            bif.sweep_max,
            bif.sweep_steps,
            variable=bif.variable,
        )
        result = _attach_analysis_metadata("bifurcation", bif.model_dump(), sweep)
        return dict(result)
    if analysis == "heatmap":
        heat = HeatmapRequest.model_validate(payload_dump)
        sim_fn = _make_simulate_fn(heat.model_dump())
        base_cfg = {
            "params": heat.params,
            "init": heat.init,
            "dt": heat.dt,
            "duration": heat.duration,
            "current": heat.current,
            "protocol": "constant",
        }
        heat_payload = heatmap_2d(
            sim_fn,
            base_cfg,
            heat.param_x,
            heat.x_min,
            heat.x_max,
            heat.x_steps,
            heat.param_y,
            heat.y_min,
            heat.y_max,
            heat.y_steps,
        )
        result = _attach_analysis_metadata("heatmap", heat.model_dump(), heat_payload)
        return dict(result)
    sens = SensitivityRequest.model_validate(payload_dump)
    sim_fn = _make_simulate_fn(sens.model_dump())
    param_names = list((sens.params or {}).keys())
    base_cfg = {
        "params": sens.params,
        "init": sens.init,
        "dt": sens.dt,
        "duration": sens.duration,
        "current": sens.current,
        "protocol": "constant",
    }
    sens_payload = sensitivity_analysis(sim_fn, base_cfg, param_names)
    result = _attach_analysis_metadata("sensitivity", sens.model_dump(), sens_payload)
    return dict(result)


def execute_analysis_process_task(
    job_context: StudioJobContext, payload: Mapping[str, object]
) -> dict[str, object]:
    """Validate a named worker request and run the existing analysis implementation.

    Parameters
    ----------
    job_context:
        Context supplied by the registered process worker, never serialised.
    payload:
        JSON object with ``analysis``, ``payload`` and ``parameter_order``.
        The explicit parameter-name sequence preserves stable sensitivity
        ordering across transports that sort JSON object keys.

    Returns
    -------
    dict[str, object]
        Existing public analysis result, including its evidence metadata.

    Raises
    ------
    ValueError
        The envelope or selected analysis payload is invalid.
    """
    if set(payload) != {"analysis", "payload", "parameter_order"}:
        raise AnalysisJobValidationError("invalid_analysis_payload")
    request = AnalysisJobRequest.model_validate(
        {"analysis": payload["analysis"], "payload": payload["payload"]}
    )
    analysis, normalized, _, _, _ = validate_analysis_job_request(request)
    order = payload["parameter_order"]
    params = normalized.get("params") or {}
    if (
        not isinstance(order, list)
        or not all(isinstance(name, str) for name in order)
        or len(order) != len(params)
        or set(order) != set(params)
    ):
        raise AnalysisJobValidationError("invalid_analysis_payload")
    if params:
        normalized["params"] = {name: params[name] for name in order}
    return run_analysis_job_task(analysis, normalized, job_context)


def submit_analysis_job(
    job_manager: StudioJobService,
    req: AnalysisJobRequest,
    *,
    request_id: str | None = None,
) -> dict[str, Any]:
    """Validate and submit one analysis job; return the public job receipt.

    Parameters
    ----------
    job_manager : StudioJobService
        Existing job admission and custody owner.
    req : AnalysisJobRequest
        Scientific analysis request, validated before admission.
    request_id : str or None
        Middleware-normalized HTTP trace, if submitted through an HTTP route.
        This is neither an identity assertion nor an idempotency key.

    Returns
    -------
    dict[str, Any]
        Public receipt and projected analysis work with status route.

    Raises
    ------
    AnalysisJobValidationError
        The analysis input is invalid or job admission refuses the work.
    """
    analysis, payload_dump, sim_count, duration, dt = validate_analysis_job_request(req)
    if sim_count < 1:
        raise AnalysisJobValidationError("analysis_job_empty")

    try:
        record = job_manager.submit_process_task(
            kind="analysis",
            owner="studio",
            request_id=request_id,
            task_path="sc_neurocore.studio.api.analysis_jobs:execute_analysis_process_task",
            payload={
                "analysis": analysis,
                "payload": payload_dump,
                "parameter_order": list(payload_dump.get("params") or {}),
            },
        )
    except StudioJobRejected as exc:
        raise AnalysisJobValidationError("analysis_job_rejected") from exc
    return {
        "analysis": analysis,
        "execution_mode": "async_job",
        "job": record.to_public_dict(),
        "job_id": record.job_id,
        "projected_simulations": sim_count,
        "schema_version": "studio.analysis.job.v1",
        "status_route": f"/api/studio/jobs/{record.job_id}",
        "duration_ms": duration,
        "dt_ms": dt,
    }
