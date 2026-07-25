# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio synchronous analysis cost contracts

from __future__ import annotations


import pytest

from sc_neurocore.studio.platform.analysis_limits import (
    AnalysisBudgetError,
    evaluate_analysis_cost,
    evaluate_multi_config_cost,
)


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
