# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Asynchronous Studio analysis job validation and execution

"""Validate and execute heavy analysis jobs off the HTTP request thread."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import ValidationError

from sc_neurocore.studio.analysis import (
    bifurcation_sweep,
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
    SensitivityRequest,
)
from sc_neurocore.studio.platform.jobs_context import StudioJobContext
from sc_neurocore.studio.platform.jobs_manager import StudioJobManager
from sc_neurocore.studio.platform.jobs_models import StudioJobRejected

AnalysisKind = Literal["fi_curve", "bifurcation", "heatmap", "sensitivity"]


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

    if analysis == "fi_curve":
        import numpy as np

        fi = FICurveRequest.model_validate(payload_dump)
        sim_fn = _make_simulate_fn(fi.model_dump())
        currents = np.linspace(fi.i_min, fi.i_max, fi.i_steps).tolist()
        rates = [sim_fn(current=float(i))["stats"]["rate_hz"] for i in currents]
        result = _attach_analysis_metadata(
            "fi_curve", fi.model_dump(), {"currents": currents, "rates": rates}
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


def submit_analysis_job(
    job_manager: StudioJobManager,
    req: AnalysisJobRequest,
) -> dict[str, Any]:
    """Validate and submit one analysis job; return the public job receipt."""

    analysis, payload_dump, sim_count, duration, dt = validate_analysis_job_request(req)
    if sim_count < 1:
        raise AnalysisJobValidationError("analysis_job_empty")

    def _task(job_context: StudioJobContext) -> dict[str, object]:
        return run_analysis_job_task(analysis, payload_dump, job_context)

    try:
        record = job_manager.submit(
            kind="analysis",
            owner="studio",
            request_id=None,
            task=_task,
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
