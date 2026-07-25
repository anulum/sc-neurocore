# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio analysis budget value contracts

from __future__ import annotations


import pytest

from sc_neurocore.studio.platform.analysis_limits import (
    AnalysisBudget,
    AnalysisBudgetError,
    AnalysisCost,
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
