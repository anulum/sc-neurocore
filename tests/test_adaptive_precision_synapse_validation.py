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
    SynapsePrecision,
    assign_synapse_precisions,
)


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


def test_assign_synapse_precisions_rejects_mismatched_sensitivity_shape() -> None:
    """Synapse planning requires each sensitivity map to match its weight layer."""
    with pytest.raises(ValueError, match="match its layer weight shape"):
        assign_synapse_precisions(
            [np.ones((2, 2))],
            sensitivity_maps=[np.ones((3, 3))],
        )
