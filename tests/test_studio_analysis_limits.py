# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio synchronous analysis budget tests

"""Tests for the synchronous Studio analysis execution budget."""

from __future__ import annotations

import math

import pytest

from sc_neurocore.studio.platform.analysis_limits import (
    STUDIO_ANALYSIS_REFERENCE_TIMESTEP_MS,
    AnalysisBudget,
    AnalysisBudgetError,
    AnalysisCost,
    enforce_analysis_budget,
    evaluate_analysis_cost,
    evaluate_multi_config_cost,
    resolve_request_timestep,
    simulation_step_count,
)


def test_analysis_budget_rejects_non_positive_ceilings() -> None:
    with pytest.raises(ValueError, match="steps-per-simulation budget"):
        AnalysisBudget(max_steps_per_simulation=0)
    with pytest.raises(ValueError, match="total-steps budget"):
        AnalysisBudget(max_total_steps=0)
    with pytest.raises(ValueError, match="simulation-count budget"):
        AnalysisBudget(max_simulations=-1)


def test_analysis_budget_public_dict_is_sorted_scalar() -> None:
    budget = AnalysisBudget(
        max_steps_per_simulation=10,
        max_total_steps=100,
        max_simulations=5,
    )

    assert budget.to_public_dict() == {
        "max_simulations": 5,
        "max_steps_per_simulation": 10,
        "max_total_steps": 100,
    }


def test_analysis_cost_public_dict() -> None:
    cost = AnalysisCost(simulation_count=3, steps_per_simulation=200, total_steps=600)

    assert cost.to_public_dict() == {
        "simulation_count": 3,
        "steps_per_simulation": 200,
        "total_steps": 600,
    }


def test_simulation_step_count_rounds_up() -> None:
    assert simulation_step_count(200.0, 0.1) == 2000
    assert simulation_step_count(10.0, 3.0) == 4  # ceil(3.33)


@pytest.mark.parametrize(
    "duration,dt",
    [
        (0.0, 0.1),
        (-1.0, 0.1),
        (100.0, 0.0),
        (100.0, -0.1),
        (math.inf, 0.1),
        (100.0, math.inf),
        (math.nan, 0.1),
        (100.0, math.nan),
    ],
)
def test_simulation_step_count_rejects_invalid_timestep(duration: float, dt: float) -> None:
    with pytest.raises(AnalysisBudgetError) as excinfo:
        simulation_step_count(duration, dt)

    assert excinfo.value.limit == "timestep"


def test_resolve_request_timestep_defaults_only_for_none() -> None:
    assert resolve_request_timestep(None) == STUDIO_ANALYSIS_REFERENCE_TIMESTEP_MS
    assert resolve_request_timestep(0.05) == 0.05
    # A supplied non-positive dt is returned unchanged so the gate can reject it.
    assert resolve_request_timestep(0.0) == 0.0
    assert resolve_request_timestep(-0.2) == -0.2


def test_evaluate_analysis_cost_scales_with_count() -> None:
    cost = evaluate_analysis_cost(simulation_count=4, duration=100.0, dt=0.1)

    assert cost.steps_per_simulation == 1000
    assert cost.simulation_count == 4
    assert cost.total_steps == 4000


def test_evaluate_analysis_cost_rejects_non_positive_count() -> None:
    with pytest.raises(AnalysisBudgetError) as excinfo:
        evaluate_analysis_cost(simulation_count=0, duration=100.0, dt=0.1)

    assert excinfo.value.limit == "simulations"


def test_evaluate_multi_config_cost_uses_max_and_sum() -> None:
    cost = evaluate_multi_config_cost([(100.0, 0.1), (50.0, 0.5)])

    assert cost.simulation_count == 2
    assert cost.steps_per_simulation == 1000  # max(1000, 100)
    assert cost.total_steps == 1100  # 1000 + 100


def test_evaluate_multi_config_cost_rejects_empty() -> None:
    with pytest.raises(AnalysisBudgetError) as excinfo:
        evaluate_multi_config_cost([])

    assert excinfo.value.limit == "simulations"


def test_evaluate_multi_config_cost_propagates_timestep_error() -> None:
    with pytest.raises(AnalysisBudgetError) as excinfo:
        evaluate_multi_config_cost([(100.0, 0.1), (100.0, 0.0)])

    assert excinfo.value.limit == "timestep"


def test_enforce_analysis_budget_allows_within_limits() -> None:
    budget = AnalysisBudget(
        max_steps_per_simulation=10_000,
        max_total_steps=100_000,
        max_simulations=100,
    )
    cost = evaluate_analysis_cost(simulation_count=10, duration=100.0, dt=0.1)

    enforce_analysis_budget(cost, budget)


def test_enforce_analysis_budget_rejects_too_many_simulations() -> None:
    budget = AnalysisBudget(max_simulations=3)
    cost = AnalysisCost(simulation_count=4, steps_per_simulation=10, total_steps=40)

    with pytest.raises(AnalysisBudgetError) as excinfo:
        enforce_analysis_budget(cost, budget)

    assert excinfo.value.limit == "simulations"
    assert excinfo.value.projected == 4
    assert excinfo.value.allowed == 3


def test_enforce_analysis_budget_rejects_per_simulation_steps() -> None:
    budget = AnalysisBudget(max_steps_per_simulation=100, max_simulations=10)
    cost = AnalysisCost(simulation_count=1, steps_per_simulation=500, total_steps=500)

    with pytest.raises(AnalysisBudgetError) as excinfo:
        enforce_analysis_budget(cost, budget)

    assert excinfo.value.limit == "steps_per_simulation"
    assert excinfo.value.projected == 500
    assert excinfo.value.allowed == 100


def test_enforce_analysis_budget_rejects_total_steps() -> None:
    budget = AnalysisBudget(
        max_steps_per_simulation=1_000,
        max_total_steps=1_500,
        max_simulations=100,
    )
    cost = AnalysisCost(simulation_count=2, steps_per_simulation=1_000, total_steps=2_000)

    with pytest.raises(AnalysisBudgetError) as excinfo:
        enforce_analysis_budget(cost, budget)

    assert excinfo.value.limit == "total_steps"
    assert excinfo.value.projected == 2_000
    assert excinfo.value.allowed == 1_500


def test_enforce_analysis_budget_checks_simulations_before_steps() -> None:
    budget = AnalysisBudget(
        max_steps_per_simulation=100,
        max_total_steps=100,
        max_simulations=1,
    )
    cost = AnalysisCost(simulation_count=2, steps_per_simulation=500, total_steps=1_000)

    with pytest.raises(AnalysisBudgetError) as excinfo:
        enforce_analysis_budget(cost, budget)

    assert excinfo.value.limit == "simulations"


def test_analysis_budget_error_detail_is_path_free() -> None:
    error = AnalysisBudgetError(
        limit="total_steps",
        projected=2_000,
        allowed=1_000,
        message="Analysis request exceeds the synchronous integration-step budget.",
    )

    detail = error.to_public_detail()

    assert detail == {
        "allowed": 1_000,
        "limit": "total_steps",
        "projected": 2_000,
        "reason": "Analysis request exceeds the synchronous integration-step budget.",
    }
    reason = detail["reason"]
    assert isinstance(reason, str)
    assert "/home/" not in reason
    assert "/media/" not in reason
