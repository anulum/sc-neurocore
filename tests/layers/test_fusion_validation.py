# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SC fusion input validation tests

"""Length, emptiness, rank, weighting, and shared-feature contracts."""

import numpy as np
import pytest

from sc_neurocore.layers.fusion import SCFusionLayer


def test_fusion_input_length_mismatch_raises() -> None:
    """Mismatched input lengths should raise a ValueError."""
    layer = SCFusionLayer(input_dims={"a": 2, "b": 3}, fusion_weights={"a": 1.0, "b": 1.0})
    with pytest.raises(ValueError):
        _ = layer.forward({"a": np.array([1.0, 1.0]), "b": np.array([1.0, 1.0, 1.0, 1.0])})


def test_fusion_empty_inputs_raise_value_error() -> None:
    """Forward requires at least one declared modality input."""
    layer = SCFusionLayer(input_dims={"a": 2}, fusion_weights={"a": 1.0})
    with pytest.raises(ValueError, match="at least one"):
        _ = layer.forward({})


def test_fusion_rejects_non_vector_input() -> None:
    """Weighted modality inputs must be one-dimensional arrays."""
    layer = SCFusionLayer(input_dims={"a": 2}, fusion_weights={"a": 1.0})
    with pytest.raises(ValueError, match="one-dimensional"):
        _ = layer.forward({"a": np.ones((1, 2))})


def test_fusion_rejects_only_unweighted_inputs() -> None:
    """Forward requires at least one input with a configured weight."""
    layer = SCFusionLayer(input_dims={"a": 2, "b": 2}, fusion_weights={"a": 1.0})
    with pytest.raises(ValueError, match="weighted modality"):
        _ = layer.forward({"b": np.ones(2)})


def test_fusion_weighted_modalities_share_feature_length() -> None:
    """Weighted modalities must resolve to the same fused feature length."""
    layer = SCFusionLayer(input_dims={"a": 2, "b": 3}, fusion_weights={"a": 1.0, "b": 1.0})
    with pytest.raises(ValueError, match="share feature length"):
        _ = layer.forward({"a": np.ones(2), "b": np.ones(3)})
