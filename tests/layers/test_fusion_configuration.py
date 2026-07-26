# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SC fusion configuration tests

"""Weight normalization, fallback, and constructor validation contracts."""

import numpy as np
import pytest

from sc_neurocore.layers.fusion import SCFusionLayer


def test_fusion_normalizes_weights() -> None:
    """Weights should normalize to sum to 1."""
    layer = SCFusionLayer(input_dims={"a": 2, "b": 2}, fusion_weights={"a": 2.0, "b": 1.0})
    total = sum(layer.norm_weights.values())
    assert np.isclose(total, 1.0)


def test_fusion_zero_weight_sum_uses_equal_weights() -> None:
    """Zero total weight falls back to equal modality weights."""
    layer = SCFusionLayer(input_dims={"a": 2, "b": 2}, fusion_weights={"a": 0.0, "b": 0.0})
    assert layer.norm_weights == {"a": 0.5, "b": 0.5}
    out = layer.forward({"a": np.array([1.0, 0.0]), "b": np.array([0.0, 1.0])})
    assert np.allclose(out, np.array([0.5, 0.5]))


def test_fusion_rejects_empty_input_dimensions() -> None:
    """Constructor requires at least one declared input dimension."""
    with pytest.raises(ValueError, match="input_dims"):
        _ = SCFusionLayer(input_dims={}, fusion_weights={"a": 1.0})


def test_fusion_rejects_empty_fusion_weights() -> None:
    """Constructor requires at least one weighted modality."""
    with pytest.raises(ValueError, match="fusion_weights"):
        _ = SCFusionLayer(input_dims={"a": 2}, fusion_weights={})


def test_fusion_rejects_non_positive_input_dimension() -> None:
    """Constructor requires positive feature counts."""
    with pytest.raises(ValueError, match="positive"):
        _ = SCFusionLayer(input_dims={"a": 0}, fusion_weights={"a": 1.0})


def test_fusion_rejects_undeclared_weighted_modality() -> None:
    """Constructor rejects weights for undeclared modalities."""
    with pytest.raises(ValueError, match="undeclared"):
        _ = SCFusionLayer(input_dims={"a": 2}, fusion_weights={"b": 1.0})
