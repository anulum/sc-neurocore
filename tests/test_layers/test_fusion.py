"""Tests for SCFusionLayer fusion logic and normalization."""

import os
import time

import numpy as np
import pytest

from sc_neurocore.layers.fusion import SCFusionLayer


def _perf_enabled() -> bool:
    return os.environ.get("SC_NEUROCORE_PERF") == "1"


def test_fusion_normalizes_weights():
    """Weights should normalize to sum to 1."""
    layer = SCFusionLayer(input_dims={"a": 2, "b": 2}, fusion_weights={"a": 2.0, "b": 1.0})
    total = sum(layer.norm_weights.values())
    assert np.isclose(total, 1.0)


def test_fusion_forward_two_modalities():
    """Fusion output matches weighted sum of inputs."""
    layer = SCFusionLayer(input_dims={"a": 2, "b": 2}, fusion_weights={"a": 1.0, "b": 1.0})
    out = layer.forward({"a": np.array([1.0, 0.0]), "b": np.array([0.0, 1.0])})
    assert np.allclose(out, np.array([0.5, 0.5]))


def test_fusion_forward_single_modality():
    """Single-modality fusion returns the input scaled by weight."""
    layer = SCFusionLayer(input_dims={"a": 3}, fusion_weights={"a": 2.0})
    out = layer.forward({"a": np.array([0.2, 0.4, 0.6])})
    assert np.allclose(out, np.array([0.2, 0.4, 0.6]))


def test_fusion_ignores_unweighted_modality():
    """Modalities not in fusion_weights are skipped."""
    layer = SCFusionLayer(input_dims={"a": 2, "b": 2}, fusion_weights={"a": 1.0})
    out = layer.forward({"a": np.array([1.0, 1.0]), "b": np.array([10.0, 10.0])})
    assert np.allclose(out, np.array([1.0, 1.0]))


def test_fusion_missing_weight_modality_skipped():
    """Missing weights should not affect the fused output."""
    layer = SCFusionLayer(input_dims={"a": 2, "b": 2}, fusion_weights={"b": 1.0})
    out = layer.forward({"a": np.array([5.0, 5.0]), "b": np.array([1.0, 1.0])})
    assert np.allclose(out, np.array([1.0, 1.0]))


def test_fusion_zero_weight_sum_raises():
    """Zero total weight should raise ZeroDivisionError."""
    with pytest.raises(ZeroDivisionError):
        _ = SCFusionLayer(input_dims={"a": 1}, fusion_weights={"a": 0.0})


def test_fusion_input_length_mismatch_raises():
    """Mismatched input lengths should raise a ValueError."""
    layer = SCFusionLayer(input_dims={"a": 2, "b": 3}, fusion_weights={"a": 1.0, "b": 1.0})
    with pytest.raises(ValueError):
        _ = layer.forward({"a": np.array([1.0, 1.0]), "b": np.array([1.0, 1.0, 1.0, 1.0])})


def test_fusion_output_shape_matches_input():
    """Output shape should match input feature dimension."""
    layer = SCFusionLayer(input_dims={"a": 4, "b": 4}, fusion_weights={"a": 1.0, "b": 1.0})
    out = layer.forward({"a": np.ones(4), "b": np.zeros(4)})
    assert out.shape == (4,)


def test_fusion_output_dtype_float():
    """Output should be float array."""
    layer = SCFusionLayer(input_dims={"a": 2}, fusion_weights={"a": 1.0})
    out = layer.forward({"a": np.array([1, 2])})
    assert np.issubdtype(out.dtype, np.floating)


@pytest.mark.skipif(not _perf_enabled(), reason="Set SC_NEUROCORE_PERF=1 to enable perf checks.")
def test_fusion_perf_small():
    """Benchmark a small fusion call."""
    layer = SCFusionLayer(input_dims={"a": 64, "b": 64}, fusion_weights={"a": 1.0, "b": 1.0})
    data = {"a": np.random.random(64), "b": np.random.random(64)}
    start = time.perf_counter()
    _ = layer.forward(data)
    elapsed = time.perf_counter() - start
    assert elapsed < 1.5
