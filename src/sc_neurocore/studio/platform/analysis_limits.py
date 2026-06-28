# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio synchronous analysis execution budget

"""Fail-closed budgets for synchronous SC-NeuroCore Studio analysis requests.

Studio analysis endpoints (single simulations, parameter sweeps, heatmaps,
sensitivity, characterisation) run synchronously inside the FastAPI request
worker. The number of integration steps a request drives is
``simulation_count * ceil(duration / dt)``; both ``duration / dt`` and the
sweep/parameter counts are caller-supplied, so an unbounded request can starve
the worker. This module projects the integration cost of a request and rejects
it before any work runs when it exceeds a configured ceiling, keeping
synchronous analysis bounded and the request worker responsive.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import ceil, isfinite
from typing import Literal, TypeAlias

DEFAULT_STUDIO_MAX_SYNC_ANALYSIS_STEPS_PER_SIMULATION = 5_000_000
DEFAULT_STUDIO_MAX_SYNC_ANALYSIS_TOTAL_STEPS = 200_000_000
DEFAULT_STUDIO_MAX_SYNC_ANALYSIS_SIMULATIONS = 4_096

STUDIO_ANALYSIS_REFERENCE_TIMESTEP_MS = 0.1
"""Reference timestep (ms) used to project cost when a request defers ``dt`` to the model default.

Model-based requests may omit ``dt`` and let the named model choose its own
default (``0.1`` ms for the built-in models). Cost projection still needs a
concrete timestep, so this reference is substituted for a deferred ``dt`` only;
an explicit non-positive ``dt`` is left untouched so it fails the timestep gate.
"""

AnalysisBudgetLimit: TypeAlias = Literal[
    "timestep",
    "steps_per_simulation",
    "total_steps",
    "simulations",
]


class AnalysisBudgetError(ValueError):
    """Raised when a synchronous analysis request exceeds its execution budget.

    Parameters
    ----------
    limit:
        Which budget dimension was violated.
    projected:
        Projected value for the violated dimension (integration steps or
        simulation count) computed from the request.
    allowed:
        Maximum value permitted for the violated dimension.
    message:
        Operator-facing summary that must not include local paths or secrets.
    """

    def __init__(
        self,
        *,
        limit: AnalysisBudgetLimit,
        projected: int,
        allowed: int,
        message: str,
    ) -> None:
        super().__init__(message)
        self.limit = limit
        self.projected = projected
        self.allowed = allowed

    def to_public_detail(self) -> dict[str, int | str]:
        """Return a JSON-serializable, path-free HTTP error detail payload."""

        return {
            "allowed": self.allowed,
            "limit": self.limit,
            "projected": self.projected,
            "reason": str(self),
        }


@dataclass(frozen=True, slots=True)
class AnalysisBudget:
    """Ceilings that bound synchronous Studio analysis execution.

    Parameters
    ----------
    max_steps_per_simulation:
        Maximum integration steps (``ceil(duration / dt)``) for any single
        simulation in a request.
    max_total_steps:
        Maximum summed integration steps across every simulation a request
        drives (``simulation_count * steps_per_simulation`` for a sweep).
    max_simulations:
        Maximum number of simulations a single request may drive, independent
        of per-simulation cost.
    """

    max_steps_per_simulation: int = DEFAULT_STUDIO_MAX_SYNC_ANALYSIS_STEPS_PER_SIMULATION
    max_total_steps: int = DEFAULT_STUDIO_MAX_SYNC_ANALYSIS_TOTAL_STEPS
    max_simulations: int = DEFAULT_STUDIO_MAX_SYNC_ANALYSIS_SIMULATIONS

    def __post_init__(self) -> None:
        """Validate that every budget ceiling is a positive integer."""

        if self.max_steps_per_simulation <= 0:
            raise ValueError("Studio analysis steps-per-simulation budget must be positive.")
        if self.max_total_steps <= 0:
            raise ValueError("Studio analysis total-steps budget must be positive.")
        if self.max_simulations <= 0:
            raise ValueError("Studio analysis simulation-count budget must be positive.")

    def to_public_dict(self) -> dict[str, int]:
        """Return a JSON-serializable, path-free budget payload."""

        return {
            "max_simulations": self.max_simulations,
            "max_steps_per_simulation": self.max_steps_per_simulation,
            "max_total_steps": self.max_total_steps,
        }


@dataclass(frozen=True, slots=True)
class AnalysisCost:
    """Projected synchronous integration cost of an analysis request.

    Parameters
    ----------
    simulation_count:
        Number of simulations the request drives.
    steps_per_simulation:
        Largest single-simulation integration-step count among the request's
        simulations.
    total_steps:
        Summed integration steps across every simulation in the request.
    """

    simulation_count: int
    steps_per_simulation: int
    total_steps: int

    def to_public_dict(self) -> dict[str, int]:
        """Return a JSON-serializable, path-free cost payload."""

        return {
            "simulation_count": self.simulation_count,
            "steps_per_simulation": self.steps_per_simulation,
            "total_steps": self.total_steps,
        }


def simulation_step_count(duration: float, dt: float) -> int:
    """Return the integration-step count for one simulation.

    Parameters
    ----------
    duration:
        Simulated time span in milliseconds. Must be finite and positive.
    dt:
        Integration timestep in milliseconds. Must be finite and positive.

    Returns
    -------
    int
        ``ceil(duration / dt)`` integration steps.

    Raises
    ------
    AnalysisBudgetError
        If ``duration`` or ``dt`` is non-finite or non-positive. The error uses
        the ``"timestep"`` limit so callers can surface a path-free 422.
    """

    if not isfinite(duration) or not isfinite(dt) or duration <= 0.0 or dt <= 0.0:
        raise AnalysisBudgetError(
            limit="timestep",
            projected=0,
            allowed=0,
            message="Analysis duration and timestep must be finite and positive.",
        )
    return ceil(duration / dt)


def resolve_request_timestep(dt: float | None) -> float:
    """Return the timestep used to project a request's cost.

    Parameters
    ----------
    dt:
        Caller-supplied timestep in milliseconds, or ``None`` when the request
        defers to the named model's default timestep.

    Returns
    -------
    float
        ``dt`` when supplied, otherwise
        :data:`STUDIO_ANALYSIS_REFERENCE_TIMESTEP_MS`. A supplied non-positive
        ``dt`` is returned unchanged so the timestep gate rejects it.
    """

    if dt is None:
        return STUDIO_ANALYSIS_REFERENCE_TIMESTEP_MS
    return dt


def evaluate_analysis_cost(
    *,
    simulation_count: int,
    duration: float,
    dt: float,
) -> AnalysisCost:
    """Project the cost of a request whose simulations share one duration/dt.

    Parameters
    ----------
    simulation_count:
        Number of simulations the request drives. Must be positive.
    duration:
        Shared simulated time span in milliseconds.
    dt:
        Shared integration timestep in milliseconds.

    Returns
    -------
    AnalysisCost
        Projected per-simulation and total integration-step counts.

    Raises
    ------
    AnalysisBudgetError
        If ``simulation_count`` is non-positive (``"simulations"`` limit) or the
        timestep is invalid (``"timestep"`` limit).
    """

    if simulation_count <= 0:
        raise AnalysisBudgetError(
            limit="simulations",
            projected=simulation_count,
            allowed=1,
            message="Analysis simulation count must be positive.",
        )
    steps = simulation_step_count(duration, dt)
    return AnalysisCost(
        simulation_count=simulation_count,
        steps_per_simulation=steps,
        total_steps=steps * simulation_count,
    )


def evaluate_multi_config_cost(configs: Sequence[tuple[float, float]]) -> AnalysisCost:
    """Project the cost of a request whose simulations have distinct duration/dt.

    Parameters
    ----------
    configs:
        One ``(duration, dt)`` pair per simulation the request drives.

    Returns
    -------
    AnalysisCost
        ``steps_per_simulation`` is the largest single-simulation cost and
        ``total_steps`` is the summed cost across ``configs``.

    Raises
    ------
    AnalysisBudgetError
        If ``configs`` is empty (``"simulations"`` limit) or any pair has an
        invalid timestep (``"timestep"`` limit).
    """

    if not configs:
        raise AnalysisBudgetError(
            limit="simulations",
            projected=0,
            allowed=1,
            message="Analysis simulation count must be positive.",
        )
    per_simulation = [simulation_step_count(duration, dt) for duration, dt in configs]
    return AnalysisCost(
        simulation_count=len(configs),
        steps_per_simulation=max(per_simulation),
        total_steps=sum(per_simulation),
    )


def evaluate_nullcline_grid_cost(*, grid_size: int, equation_count: int) -> AnalysisCost:
    """Project the synchronous point-evaluation cost of a nullcline grid.

    Parameters
    ----------
    grid_size:
        Number of points per axis in the square nullcline grid.
    equation_count:
        Number of ODE right-hand-side expressions evaluated at each grid point.

    Returns
    -------
    AnalysisCost
        Cost projection where each grid point is treated as one synchronous
        unit of work and ``steps_per_simulation`` records the equation
        evaluations performed at that point.

    Raises
    ------
    AnalysisBudgetError
        If either count is non-positive. The ``"simulations"`` limit is used
        because the grid point count is checked against the same synchronous
        request fan-out ceiling as parameter sweeps.
    """

    if grid_size <= 0 or equation_count <= 0:
        raise AnalysisBudgetError(
            limit="simulations",
            projected=0,
            allowed=1,
            message="Nullcline grid size and equation count must be positive.",
        )
    grid_points = grid_size * grid_size
    return AnalysisCost(
        simulation_count=grid_points,
        steps_per_simulation=equation_count,
        total_steps=grid_points * equation_count,
    )


def enforce_analysis_budget(cost: AnalysisCost, budget: AnalysisBudget) -> None:
    """Reject a projected analysis cost that exceeds the configured budget.

    Parameters
    ----------
    cost:
        Projected request cost from :func:`evaluate_analysis_cost` or
        :func:`evaluate_multi_config_cost`.
    budget:
        Active synchronous analysis ceilings.

    Raises
    ------
    AnalysisBudgetError
        If the simulation count, per-simulation steps, or total steps exceed the
        budget. The first violated dimension is reported, in
        simulations -> per-simulation -> total order.
    """

    if cost.simulation_count > budget.max_simulations:
        raise AnalysisBudgetError(
            limit="simulations",
            projected=cost.simulation_count,
            allowed=budget.max_simulations,
            message=(
                "Analysis request drives too many simulations for synchronous execution; "
                "reduce the sweep, grid, or parameter count."
            ),
        )
    if cost.steps_per_simulation > budget.max_steps_per_simulation:
        raise AnalysisBudgetError(
            limit="steps_per_simulation",
            projected=cost.steps_per_simulation,
            allowed=budget.max_steps_per_simulation,
            message=(
                "A single simulation exceeds the synchronous step budget; "
                "shorten the duration or increase the timestep."
            ),
        )
    if cost.total_steps > budget.max_total_steps:
        raise AnalysisBudgetError(
            limit="total_steps",
            projected=cost.total_steps,
            allowed=budget.max_total_steps,
            message=(
                "Analysis request exceeds the synchronous integration-step budget; "
                "reduce the sweep size, duration, or timestep resolution."
            ),
        )


__all__ = [
    "DEFAULT_STUDIO_MAX_SYNC_ANALYSIS_SIMULATIONS",
    "DEFAULT_STUDIO_MAX_SYNC_ANALYSIS_STEPS_PER_SIMULATION",
    "DEFAULT_STUDIO_MAX_SYNC_ANALYSIS_TOTAL_STEPS",
    "STUDIO_ANALYSIS_REFERENCE_TIMESTEP_MS",
    "AnalysisBudget",
    "AnalysisBudgetError",
    "AnalysisBudgetLimit",
    "AnalysisCost",
    "enforce_analysis_budget",
    "evaluate_analysis_cost",
    "evaluate_multi_config_cost",
    "evaluate_nullcline_grid_cost",
    "resolve_request_timestep",
    "simulation_step_count",
]
