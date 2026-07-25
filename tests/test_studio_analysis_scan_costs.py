# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio nullcline and catalogue scan cost contracts

from __future__ import annotations


import pytest

from sc_neurocore.studio.platform.analysis_limits import (
    AnalysisBudgetError,
    evaluate_model_scan_cost,
    evaluate_nullcline_grid_cost,
)


def test_evaluate_nullcline_grid_cost_counts_grid_points_as_simulations() -> None:
    cost = evaluate_nullcline_grid_cost(grid_size=60, equation_count=2)

    assert cost.simulation_count == 3_600
    assert cost.steps_per_simulation == 2
    assert cost.total_steps == 7_200


@pytest.mark.parametrize(
    ("grid_size", "equation_count"),
    [
        (0, 2),
        (60, 0),
    ],
)
def test_evaluate_nullcline_grid_cost_rejects_invalid_counts(
    grid_size: int,
    equation_count: int,
) -> None:
    with pytest.raises(AnalysisBudgetError) as excinfo:
        evaluate_nullcline_grid_cost(grid_size=grid_size, equation_count=equation_count)

    assert excinfo.value.limit == "simulations"


def test_evaluate_model_scan_cost_counts_catalogue_runs() -> None:
    """Model scan cost is one bounded simulation per catalogue model."""

    cost = evaluate_model_scan_cost(
        model_count=12,
        duration=100.0,
        dt=0.1,
    )

    assert cost.to_public_dict() == {
        "simulation_count": 12,
        "steps_per_simulation": 1_000,
        "total_steps": 12_000,
    }


@pytest.mark.parametrize("model_count", [0, -1])
def test_evaluate_model_scan_cost_rejects_empty_catalogues(model_count: int) -> None:
    """The model-scan guard fails closed if the catalogue count is invalid."""

    with pytest.raises(AnalysisBudgetError) as excinfo:
        evaluate_model_scan_cost(
            model_count=model_count,
            duration=100.0,
            dt=0.1,
        )

    assert excinfo.value.limit == "simulations"
    assert excinfo.value.projected == model_count
    assert "catalogue" in str(excinfo.value)
