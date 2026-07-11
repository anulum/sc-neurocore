# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio synchronous analysis guards

"""Project and enforce synchronous Studio analysis costs."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import HTTPException

from sc_neurocore.studio.analysis_manifest import (
    attach_analysis_result_manifest,
    infer_analysis_source,
)
from sc_neurocore.studio.models import simulate_model
from sc_neurocore.studio.platform import (
    AnalysisBudget,
    AnalysisBudgetError,
    StudioRuntimeSettings,
    enforce_analysis_budget,
    evaluate_analysis_cost,
    evaluate_model_scan_cost,
    evaluate_multi_config_cost,
    evaluate_nullcline_grid_cost,
    resolve_request_timestep,
)
from sc_neurocore.studio.simulation import simulate


def _analysis_budget_from_settings(settings: StudioRuntimeSettings) -> AnalysisBudget:
    """Build the synchronous analysis execution budget from runtime settings."""
    return AnalysisBudget(
        max_steps_per_simulation=settings.max_sync_analysis_steps_per_simulation,
        max_total_steps=settings.max_sync_analysis_total_steps,
        max_simulations=settings.max_sync_analysis_simulations,
    )


def _guard_analysis_request(
    budget: AnalysisBudget,
    *,
    simulation_count: int,
    duration: float,
    dt: float | None,
) -> None:
    """Reject an analysis request whose projected synchronous cost is over budget.

    Parameters
    ----------
    budget:
        Active synchronous analysis ceilings.
    simulation_count:
        Number of simulations the request drives.
    duration:
        Shared simulated time span in milliseconds.
    dt:
        Shared timestep in milliseconds, or ``None`` when the request defers to
        the model default.

    Raises
    ------
    HTTPException
        With status 422 and a path-free budget detail when the request exceeds
        the configured synchronous analysis budget.
    """
    try:
        cost = evaluate_analysis_cost(
            simulation_count=simulation_count,
            duration=duration,
            dt=resolve_request_timestep(dt),
        )
        enforce_analysis_budget(cost, budget)
    except AnalysisBudgetError as exc:
        raise HTTPException(status_code=422, detail=exc.to_public_detail()) from None


def _guard_multi_config_analysis_request(
    budget: AnalysisBudget,
    configs: list[tuple[float, float | None]],
) -> None:
    """Reject a multi-config analysis request whose projected cost is over budget.

    Parameters
    ----------
    budget:
        Active synchronous analysis ceilings.
    configs:
        One ``(duration, dt)`` pair per simulation; ``dt`` may be ``None`` when
        the simulation defers to the model default.

    Raises
    ------
    HTTPException
        With status 422 and a path-free budget detail when the request exceeds
        the configured synchronous analysis budget.
    """
    try:
        cost = evaluate_multi_config_cost(
            [(duration, resolve_request_timestep(dt)) for duration, dt in configs]
        )
        enforce_analysis_budget(cost, budget)
    except AnalysisBudgetError as exc:
        raise HTTPException(status_code=422, detail=exc.to_public_detail()) from None


def _guard_nullcline_grid_request(
    budget: AnalysisBudget,
    *,
    grid_size: int,
    equation_count: int,
) -> None:
    """Reject a nullcline grid that exceeds the synchronous analysis budget.

    Parameters
    ----------
    budget:
        Active synchronous analysis ceilings.
    grid_size:
        Number of points per axis in the requested nullcline grid.
    equation_count:
        Number of ODE right-hand-side expressions evaluated at each grid point.

    Raises
    ------
    HTTPException
        With status 422 and a path-free budget detail when the grid exceeds the
        configured synchronous analysis budget.
    """
    try:
        cost = evaluate_nullcline_grid_cost(
            grid_size=grid_size,
            equation_count=equation_count,
        )
        enforce_analysis_budget(cost, budget)
    except AnalysisBudgetError as exc:
        raise HTTPException(status_code=422, detail=exc.to_public_detail()) from None


def _guard_model_scan_request(
    budget: AnalysisBudget,
    *,
    model_count: int,
    duration: float,
) -> None:
    """Reject a catalogue model scan whose projected synchronous cost is over budget.

    Parameters
    ----------
    budget:
        Active synchronous analysis ceilings.
    model_count:
        Number of catalogue models the scan will simulate.
    duration:
        Shared model-scan duration in milliseconds.

    Raises
    ------
    HTTPException
        With status 422 and a path-free budget detail when the scan exceeds the
        configured synchronous analysis budget.
    """
    try:
        cost = evaluate_model_scan_cost(
            model_count=model_count,
            duration=duration,
            dt=resolve_request_timestep(None),
        )
        enforce_analysis_budget(cost, budget)
    except AnalysisBudgetError as exc:
        raise HTTPException(status_code=422, detail=exc.to_public_detail()) from None


def _config_duration_dt(config: dict[str, Any]) -> tuple[float, float | None]:
    """Extract a ``(duration, dt)`` cost pair from a free-form simulate config.

    Mirrors the defaults used by :func:`_make_simulate_fn` (duration ``200`` ms,
    model-default timestep when absent). Non-numeric values fall back to the
    defaults; the simulate call validates the real payload downstream.
    """
    raw_duration = config.get("duration", 200.0)
    raw_dt = config.get("dt")
    duration = float(raw_duration) if isinstance(raw_duration, (int, float)) else 200.0
    dt = float(raw_dt) if isinstance(raw_dt, (int, float)) else None
    return duration, dt


def _make_simulate_fn(req_dict: dict[str, Any]) -> Callable[..., dict[str, Any]]:
    """Build a simulate callable from request params (ODE or model)."""
    if req_dict.get("model_name"):

        def fn(**overrides: Any) -> dict[str, Any]:
            cfg = {
                "name": req_dict["model_name"],
                "param_overrides": overrides.get("params", req_dict.get("params")),
                "dt": overrides.get("dt", req_dict.get("dt")),
                "duration": overrides.get("duration", req_dict.get("duration", 200)),
                "current": overrides.get("current", req_dict.get("current", 10)),
                "protocol": overrides.get("protocol", req_dict.get("protocol", "constant")),
                "frequency_hz": overrides.get("frequency_hz", req_dict.get("frequency_hz", 10.0)),
            }
            return simulate_model(**cfg)

        return fn
    else:

        def fn(**overrides: Any) -> dict[str, Any]:
            return simulate(
                equations=req_dict.get("equations", []),
                threshold=req_dict.get("threshold"),
                reset=req_dict.get("reset"),
                params=overrides.get("params", req_dict.get("params")),
                init=overrides.get("init", req_dict.get("init")),
                dt=overrides.get("dt", req_dict.get("dt", 0.1)),
                duration=overrides.get("duration", req_dict.get("duration", 200)),
                current=overrides.get("current", req_dict.get("current", 10)),
                protocol=overrides.get("protocol", req_dict.get("protocol", "constant")),
                frequency_hz=overrides.get("frequency_hz", req_dict.get("frequency_hz", 10.0)),
            )

        return fn


def _attach_analysis_metadata(
    analysis_type: str,
    request_payload: dict[str, Any],
    result_payload: dict[str, Any],
) -> dict[str, Any]:
    """Attach path-free analysis metadata to one Studio analysis response."""
    return attach_analysis_result_manifest(
        analysis_type=analysis_type,
        source=infer_analysis_source(request_payload),
        request_payload=request_payload,
        result_payload=result_payload,
    )
