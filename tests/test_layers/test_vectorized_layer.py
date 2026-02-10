"""Tests for VectorizedSCLayer packed operations and outputs."""

import os
import time

import numpy as np
import pytest

from sc_neurocore.layers.vectorized_layer import VectorizedSCLayer


def _perf_enabled() -> bool:
    return os.environ.get("SC_NEUROCORE_PERF") == "1"


def _expected_words(length: int) -> int:
    return (length + 63) // 64


def test_vectorized_packed_shape():
    """Packed weights should have expected word dimension."""
    np.random.seed(0)
    layer = VectorizedSCLayer(n_inputs=3, n_neurons=2, length=130)
    assert layer.packed_weights.shape == (2, 3, _expected_words(130))


def test_vectorized_packed_dtype():
    """Packed weights should be uint64 for bitwise ops."""
    layer = VectorizedSCLayer(n_inputs=2, n_neurons=2, length=64)
    assert layer.packed_weights.dtype == np.uint64


def test_vectorized_forward_shape():
    """Forward returns (n_neurons,) output."""
    layer = VectorizedSCLayer(n_inputs=2, n_neurons=4, length=32)
    out = layer.forward([0.3, 0.7])
    assert out.shape == (4,)


def test_vectorized_forward_zero_input_returns_zero():
    """Zero inputs yield near-zero outputs."""
    layer = VectorizedSCLayer(n_inputs=3, n_neurons=2, length=32)
    out = layer.forward([0.0, 0.0, 0.0])
    assert np.allclose(out, 0.0)


def test_vectorized_output_range():
    """Outputs should be within [0, n_inputs]."""
    layer = VectorizedSCLayer(n_inputs=4, n_neurons=3, length=64)
    out = layer.forward([0.2, 0.4, 0.6, 0.8])
    assert np.all(out >= 0.0)
    assert np.all(out <= 4.0)


def test_vectorized_refresh_changes_packed_weights():
    """Refreshing after weight changes updates packed representation."""
    np.random.seed(1)
    layer = VectorizedSCLayer(n_inputs=2, n_neurons=2, length=32)
    before = layer.packed_weights.copy()
    layer.weights[:] = 0.0
    layer._refresh_packed_weights()
    assert not np.array_equal(before, layer.packed_weights)


def test_vectorized_deterministic_with_seed():
    """Setting numpy seed yields repeatable weights."""
    np.random.seed(99)
    layer_a = VectorizedSCLayer(n_inputs=2, n_neurons=2, length=32)
    np.random.seed(99)
    layer_b = VectorizedSCLayer(n_inputs=2, n_neurons=2, length=32)
    assert np.allclose(layer_a.weights, layer_b.weights)


def test_vectorized_input_length_mismatch_raises():
    """Input length mismatch should raise a broadcasting error."""
    layer = VectorizedSCLayer(n_inputs=3, n_neurons=2, length=16)
    with pytest.raises(ValueError):
        _ = layer.forward([0.1, 0.2])


def test_vectorized_length_not_multiple_of_64():
    """Lengths not divisible by 64 should still work."""
    layer = VectorizedSCLayer(n_inputs=1, n_neurons=1, length=70)
    out = layer.forward([0.5])
    assert out.shape == (1,)


@pytest.mark.skipif(not _perf_enabled(), reason="Set SC_NEUROCORE_PERF=1 to enable perf checks.")
def test_vectorized_layer_perf_small():
    """Benchmark a small vectorized forward pass."""
    layer = VectorizedSCLayer(n_inputs=8, n_neurons=32, length=128)
    start = time.perf_counter()
    _ = layer.forward([0.5] * 8)
    elapsed = time.perf_counter() - start
    assert elapsed < 3.0
