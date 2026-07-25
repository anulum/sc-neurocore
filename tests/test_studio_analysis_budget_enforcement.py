# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio analysis budget enforcement contracts

from __future__ import annotations


import pytest

from sc_neurocore.studio.platform.analysis_limits import (
    AnalysisBudget,
    AnalysisBudgetError,
    AnalysisCost,
    enforce_analysis_budget,
    evaluate_analysis_cost,
)


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
