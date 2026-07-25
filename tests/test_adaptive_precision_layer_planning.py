# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Adaptive precision planner contracts

"""Focused adaptive precision planner contracts."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from sc_neurocore.compiler.adaptive_precision import (
    LayerPrecision,
    assign_lengths,
)


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
