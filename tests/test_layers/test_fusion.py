# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for SCFusionLayer fusion logic and normalization

"""Tests for SCFusionLayer fusion logic and normalization."""

import os
import time

import numpy as np
import pytest

from sc_neurocore.layers.fusion import SCFusionLayer


def _perf_enabled() -> bool:
    return os.environ.get("SC_NEUROCORE_PERF") == "1"


def test_fusion_normalizes_weights() -> None:
    """Weights should normalize to sum to 1."""
    layer = SCFusionLayer(input_dims={"a": 2, "b": 2}, fusion_weights={"a": 2.0, "b": 1.0})
    total = sum(layer.norm_weights.values())
    assert np.isclose(total, 1.0)


def test_fusion_forward_two_modalities() -> None:
    """Fusion output matches weighted sum of inputs."""
    layer = SCFusionLayer(input_dims={"a": 2, "b": 2}, fusion_weights={"a": 1.0, "b": 1.0})
    out = layer.forward({"a": np.array([1.0, 0.0]), "b": np.array([0.0, 1.0])})
    assert np.allclose(out, np.array([0.5, 0.5]))


def test_fusion_forward_single_modality() -> None:
    """Single-modality fusion returns the input scaled by weight."""
    layer = SCFusionLayer(input_dims={"a": 3}, fusion_weights={"a": 2.0})
    out = layer.forward({"a": np.array([0.2, 0.4, 0.6])})
    assert np.allclose(out, np.array([0.2, 0.4, 0.6]))


def test_fusion_ignores_unweighted_modality() -> None:
    """Modalities not in fusion_weights are skipped."""
    layer = SCFusionLayer(input_dims={"a": 2, "b": 2}, fusion_weights={"a": 1.0})
    out = layer.forward({"a": np.array([1.0, 1.0]), "b": np.array([10.0, 10.0])})
    assert np.allclose(out, np.array([1.0, 1.0]))


def test_fusion_missing_weight_modality_skipped() -> None:
    """Missing weights should not affect the fused output."""
    layer = SCFusionLayer(input_dims={"a": 2, "b": 2}, fusion_weights={"b": 1.0})
    out = layer.forward({"a": np.array([5.0, 5.0]), "b": np.array([1.0, 1.0])})
    assert np.allclose(out, np.array([1.0, 1.0]))


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


def test_fusion_output_shape_matches_input() -> None:
    """Output shape should match input feature dimension."""
    layer = SCFusionLayer(input_dims={"a": 4, "b": 4}, fusion_weights={"a": 1.0, "b": 1.0})
    out = layer.forward({"a": np.ones(4), "b": np.zeros(4)})
    assert out.shape == (4,)


def test_fusion_output_dtype_float() -> None:
    """Output should be float array."""
    layer = SCFusionLayer(input_dims={"a": 2}, fusion_weights={"a": 1.0})
    out = layer.forward({"a": np.array([1, 2])})
    assert np.issubdtype(out.dtype, np.floating)


@pytest.mark.skipif(not _perf_enabled(), reason="Set SC_NEUROCORE_PERF=1 to enable perf checks.")
def test_fusion_perf_small() -> None:
    """Benchmark a small fusion call."""
    layer = SCFusionLayer(input_dims={"a": 64, "b": 64}, fusion_weights={"a": 1.0, "b": 1.0})
    data = {"a": np.random.random(64), "b": np.random.random(64)}
    start = time.perf_counter()
    _ = layer.forward(data)
    elapsed = time.perf_counter() - start
    assert elapsed < 1.5
