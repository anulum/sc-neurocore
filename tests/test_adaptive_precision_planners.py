# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Adaptive precision planner contract tests

"""Contract tests for adaptive precision planner validation."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from sc_neurocore.compiler.adaptive_precision import (
    LayerPrecision,
    SynapsePrecision,
    assign_lengths,
    assign_synapse_precisions,
    precision_plan_manifest,
)
from sc_neurocore.compiler.synapse_planner import _hoeffding_radius


def test_assign_lengths_rejects_layer_name_count_mismatch() -> None:
    """Layer planning fails closed when names cannot label every layer."""
    weights = [np.ones((2, 2)), np.ones((3, 2))]

    with pytest.raises(ValueError, match="layer_names"):
        assign_lengths(weights, layer_names=["only_one_name"])


def test_assign_lengths_rejects_unknown_method() -> None:
    """Layer planning rejects unsupported planning methods explicitly."""
    with pytest.raises(ValueError, match="method"):
        assign_lengths([np.ones((2, 2))], method="mystery")


def test_assign_lengths_rejects_empty_or_bad_rank_weights() -> None:
    """Layer planning rejects tensors that cannot describe real layer weights."""
    with pytest.raises(ValueError, match="must not be empty"):
        assign_lengths([np.array([], dtype=float)])

    with pytest.raises(ValueError, match="1D or 2D"):
        assign_lengths([np.ones((2, 2, 2))])


@pytest.mark.parametrize(
    "factory",
    [
        lambda: assign_lengths([np.ones((2, 2))], min_length=0),
        lambda: assign_lengths([np.ones((2, 2))], max_length=0),
        lambda: assign_lengths([np.ones((2, 2))], target_error=0.0),
        lambda: assign_lengths([np.ones((2, 2))], target_error=float("nan")),
    ],
)
def test_assign_lengths_rejects_invalid_scalar_bounds(
    factory: Callable[[], list[LayerPrecision]],
) -> None:
    """Layer planning validates length and error bounds before assignment."""
    with pytest.raises(ValueError, match="length bounds|target_error"):
        factory()


def test_assign_lengths_rejects_invalid_sensitivity_budget() -> None:
    """Sensitivity planning rejects non-positive explicit total budgets."""
    with pytest.raises(ValueError, match="total_budget"):
        assign_lengths([np.ones((2, 2))], method="sensitivity", total_budget=0)


def test_assign_lengths_rejects_non_finite_weights() -> None:
    """Layer planning rejects non-finite weights before sensitivity analysis."""
    with pytest.raises(ValueError, match="finite"):
        assign_lengths([np.array([[1.0, np.inf]])])


def test_assign_lengths_supports_one_dimensional_weight_vectors() -> None:
    """One-dimensional weight vectors are treated as single-output layers."""
    [row] = assign_lengths([np.array([0.1, 0.3, 0.5])], layer_names=["vector"])

    assert isinstance(row, LayerPrecision)
    assert row.name == "vector"
    assert row.layer_index == 0
    assert row.bitstream_length >= 32


def test_assign_synapse_precisions_rejects_empty_weight_layer() -> None:
    """Synapse planning fails closed for layers with no synapses."""
    with pytest.raises(ValueError, match="must not be empty"):
        assign_synapse_precisions([np.array([], dtype=float)])


def test_assign_synapse_precisions_rejects_layer_name_count_mismatch() -> None:
    """Synapse planning fails closed when names cannot label every layer."""
    weights = [np.ones((1, 2)), np.ones((1, 3))]

    with pytest.raises(ValueError, match="layer_names"):
        assign_synapse_precisions(weights, layer_names=["only_one_name"])


def test_assign_synapse_precisions_rejects_invalid_confidence() -> None:
    """Synapse planning requires confidence in the open interval (0, 1)."""
    with pytest.raises(ValueError, match="confidence"):
        assign_synapse_precisions([np.ones((1, 2))], confidence=0.0)

    with pytest.raises(ValueError, match="confidence"):
        assign_synapse_precisions([np.ones((1, 2))], confidence=1.0)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: assign_synapse_precisions([np.ones((1, 2))], target_error=0.0),
        lambda: assign_synapse_precisions([np.ones((1, 2))], target_error=float("nan")),
        lambda: assign_synapse_precisions([np.ones((1, 2))], min_bits=0),
        lambda: assign_synapse_precisions([np.ones((1, 2))], min_bits=2, max_bits=1),
        lambda: assign_synapse_precisions([np.ones((1, 2))], min_length=0),
        lambda: assign_synapse_precisions([np.ones((1, 2))], min_length=2, max_length=1),
        lambda: assign_synapse_precisions(
            [np.ones((1, 2))],
            sensitivity_maps=[np.ones((1, 2)), np.ones((1, 2))],
        ),
    ],
)
def test_assign_synapse_precisions_rejects_invalid_scalar_bounds(
    factory: Callable[[], list[SynapsePrecision]],
) -> None:
    """Synapse planning validates public scalar and list bounds early."""
    with pytest.raises(ValueError):
        factory()


def test_assign_synapse_precisions_rejects_bad_rank_weights() -> None:
    """Synapse planning rejects tensors that are not vectors or matrices."""
    with pytest.raises(ValueError, match="1D or 2D"):
        assign_synapse_precisions([np.ones((1, 2, 3))])


def test_assign_synapse_precisions_rejects_non_finite_weights() -> None:
    """Synapse planning rejects non-finite weight tensors before row emission."""
    with pytest.raises(ValueError, match="finite"):
        assign_synapse_precisions([np.array([[0.5, np.nan]])])


def test_assign_synapse_precisions_rejects_non_finite_sensitivity_maps() -> None:
    """Synapse planning rejects sensitivity maps outside finite non-negative values."""
    with pytest.raises(ValueError, match="finite non-negative"):
        assign_synapse_precisions(
            [np.ones((1, 2))],
            sensitivity_maps=[np.array([[0.1, np.inf]])],
        )


def test_zero_sensitivity_synapses_use_minimum_length_and_zero_stochastic_bound() -> None:
    """Zero-sensitivity synapses keep the minimum bitstream length."""
    [row] = assign_synapse_precisions(
        [np.array([[0.0]])],
        min_length=16,
        max_length=128,
    )

    assert row.bitstream_length == 16
    assert row.sensitivity == 0.0
    assert row.stochastic_error_bound == 0.0


def test_assign_synapse_precisions_supports_one_dimensional_sensitivity_maps() -> None:
    """One-dimensional synapse plans preserve vector coordinates in manifest rows."""
    rows = assign_synapse_precisions(
        [np.array([0.25, 0.75])],
        layer_names=["vector"],
        sensitivity_maps=[np.array([0.1, 1.0])],
        target_error=0.05,
        min_bits=2,
        max_bits=8,
        min_length=16,
        max_length=512,
    )

    assert len(rows) == 2
    assert all(isinstance(row, SynapsePrecision) for row in rows)
    assert [row.to_dict()["input_index"] for row in rows] == [0, 1]
    assert {row.layer_name for row in rows} == {"vector"}
    assert rows[1].sensitivity > rows[0].sensitivity


def test_precision_plan_manifest_handles_empty_assignment_lists() -> None:
    """The public manifest API reports zero costs for an empty precision plan."""
    manifest = precision_plan_manifest([])

    assert manifest["num_synapses"] == 0
    assert manifest["cost_summary"]["estimated_lut_cost"] == 0.0


def test_hoeffding_radius_rejects_non_positive_length() -> None:
    """The internal Hoeffding helper fails closed for invalid lengths."""
    with pytest.raises(ValueError, match="length"):
        _hoeffding_radius(0, 0.95)


def test_hoeffding_radius_rejects_invalid_confidence() -> None:
    """The internal Hoeffding helper rejects invalid confidence values."""
    with pytest.raises(ValueError, match="confidence"):
        _hoeffding_radius(16, 1.0)
