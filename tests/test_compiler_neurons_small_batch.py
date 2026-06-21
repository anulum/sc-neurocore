# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for precision solving, synapse planning and homeostatic LIF edges

"""Contracts for residual edges in precision solving, synapse planning and homeostatic LIF."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from sc_neurocore.compiler.precision_solver import (
    _min_bits_for_range,
    _min_frac_for_resolution,
    solve_precision,
)
from sc_neurocore.compiler.synapse_planner import (
    _hoeffding_radius,
    _precision_cost_summary,
    assign_synapse_precisions,
)
from sc_neurocore.neurons.homeostatic_lif import HomeostaticLIFNeuron


def test_min_bits_for_zero_range() -> None:
    """A zero-magnitude range needs a single bit."""
    assert _min_bits_for_range(0.0, 0.0) == 1


def test_min_frac_for_non_positive_resolution_defaults_high() -> None:
    """A non-positive resolution defaults to high fractional precision."""
    assert _min_frac_for_resolution(0.0) == 16


def test_solve_precision_reduces_to_fit_total_bits() -> None:
    """The solver reduces aligned fractional bits to fit a tight total-bit budget."""
    spec = solve_precision({"v": (-10.0, 10.0)}, max_total_bits=6, align_to=2)
    assert "v" in spec.var_configs


def test_homeostatic_lif_get_state_reports_threshold_and_trace() -> None:
    """HomeostaticLIFNeuron.get_state exposes the adapted threshold and rate trace."""
    neuron = HomeostaticLIFNeuron(target_rate=0.1, noise_std=0.0)
    neuron.step(1.0)

    state = neuron.get_state()

    assert "threshold" in state
    assert "rate_trace" in state


def test_hoeffding_radius_rejects_full_confidence() -> None:
    """A confidence of 1.0 leaves no failure probability and is rejected."""
    with pytest.raises(ValueError, match="confidence"):
        _hoeffding_radius(100, 1.0)


def test_precision_cost_summary_handles_no_assignments() -> None:
    """An empty assignment list yields a zero-cost summary."""
    summary = _precision_cost_summary([])
    assert summary["estimated_lut_cost"] == 0.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"target_error": 0.0},
        {"min_bits": 0},
        {"min_length": 0},
        {"layer_names": ["a", "b"]},
        {"sensitivity_maps": [np.ones((2, 2)), np.ones((2, 2))]},
    ],
)
def test_assign_synapse_precisions_rejects_invalid_arguments(kwargs: dict[str, Any]) -> None:
    """assign_synapse_precisions validates error/bit/length bounds and per-layer list lengths."""
    with pytest.raises(ValueError):
        assign_synapse_precisions([np.ones((2, 2))], **kwargs)


def test_assign_synapse_precisions_rejects_bad_rank_and_sensitivity_shape() -> None:
    """assign rejects >2-D weights and mismatched sensitivity-map shapes."""
    with pytest.raises(ValueError, match="1D or 2D"):
        assign_synapse_precisions([np.ones((2, 2, 2))])
    with pytest.raises(ValueError, match="match its layer weight shape"):
        assign_synapse_precisions([np.ones((2, 2))], sensitivity_maps=[np.ones((3, 3))])
    with pytest.raises(ValueError, match="finite non-negative"):
        assign_synapse_precisions(
            [np.ones((2, 2))], sensitivity_maps=[np.array([[-1.0, 1.0], [1.0, 1.0]])]
        )
