"""Tests for MemristiveDenseLayer defects and forward output."""

import os
import time

import numpy as np
import pytest

from sc_neurocore.layers.memristive import MemristiveDenseLayer


def _perf_enabled() -> bool:
    return os.environ.get("SC_NEUROCORE_PERF") == "1"


def test_memristive_weights_in_range():
    """Weights should remain within [0,1] after defects."""
    np.random.seed(0)
    layer = MemristiveDenseLayer(n_inputs=3, n_neurons=2, length=32, variability=0.2, stuck_rate=0.2)
    assert np.all(layer.weights >= 0.0)
    assert np.all(layer.weights <= 1.0)


def test_memristive_packed_shape():
    """Packed weights shape should match (n_neurons, n_inputs, words)."""
    layer = MemristiveDenseLayer(n_inputs=2, n_neurons=3, length=65)
    words = (65 + 63) // 64
    assert layer.packed_weights.shape == (3, 2, words)


def test_memristive_stuck_rate_one_forces_binary():
    """With stuck_rate=1, all weights become 0 or 1."""
    np.random.seed(1)
    layer = MemristiveDenseLayer(n_inputs=2, n_neurons=2, length=16, variability=0.0, stuck_rate=1.0)
    unique_vals = np.unique(layer.weights)
    assert set(unique_vals.tolist()) <= {0.0, 1.0}


def test_memristive_apply_defects_no_change_when_zero_rates():
    """Applying defects with zero rates should leave weights unchanged."""
    np.random.seed(2)
    layer = MemristiveDenseLayer(n_inputs=2, n_neurons=2, length=16, variability=0.0, stuck_rate=0.0)
    before = layer.weights.copy()
    layer.apply_hardware_defects()
    assert np.allclose(before, layer.weights)


def test_memristive_variability_increases_std():
    """Higher variability should increase weight spread."""
    np.random.seed(3)
    low_var = MemristiveDenseLayer(n_inputs=4, n_neurons=4, length=16, variability=0.0, stuck_rate=0.0)
    np.random.seed(3)
    high_var = MemristiveDenseLayer(n_inputs=4, n_neurons=4, length=16, variability=0.2, stuck_rate=0.0)
    assert np.std(high_var.weights) >= np.std(low_var.weights)


def test_memristive_forward_shape():
    """Forward returns (n_neurons,) output."""
    layer = MemristiveDenseLayer(n_inputs=2, n_neurons=5, length=32)
    out = layer.forward([0.2, 0.8])
    assert out.shape == (5,)


def test_memristive_forward_zero_input_returns_zero():
    """Zero input produces near-zero output."""
    layer = MemristiveDenseLayer(n_inputs=2, n_neurons=2, length=32)
    out = layer.forward([0.0, 0.0])
    assert np.allclose(out, 0.0)


def test_memristive_refresh_after_defects_changes_packed():
    """Packed weights should update after defects if weights change."""
    np.random.seed(4)
    layer = MemristiveDenseLayer(n_inputs=2, n_neurons=2, length=32, variability=0.0, stuck_rate=0.0)
    before = layer.packed_weights.copy()
    layer.stuck_rate = 1.0
    layer.apply_hardware_defects()
    assert not np.array_equal(before, layer.packed_weights)


def test_memristive_seed_determinism():
    """Numpy seed produces deterministic weights and defects."""
    np.random.seed(5)
    layer_a = MemristiveDenseLayer(n_inputs=2, n_neurons=2, length=16, variability=0.1, stuck_rate=0.2)
    np.random.seed(5)
    layer_b = MemristiveDenseLayer(n_inputs=2, n_neurons=2, length=16, variability=0.1, stuck_rate=0.2)
    assert np.allclose(layer_a.weights, layer_b.weights)


@pytest.mark.skipif(not _perf_enabled(), reason="Set SC_NEUROCORE_PERF=1 to enable perf checks.")
def test_memristive_perf_small():
    """Benchmark a small memristive forward pass."""
    layer = MemristiveDenseLayer(n_inputs=4, n_neurons=16, length=64)
    start = time.perf_counter()
    _ = layer.forward([0.5, 0.5, 0.5, 0.5])
    elapsed = time.perf_counter() - start
    assert elapsed < 3.0
