"""Tests for SCRecurrentLayer state updates and initialization."""

import os
import time

import numpy as np
import pytest

from sc_neurocore.layers.recurrent import SCRecurrentLayer


def _perf_enabled() -> bool:
    return os.environ.get("SC_NEUROCORE_PERF") == "1"


def test_recurrent_shapes():
    """W_in, W_rec, and state shapes should match configuration."""
    layer = SCRecurrentLayer(n_inputs=3, n_neurons=4, seed=0)
    assert layer.W_in.shape == (4, 3)
    assert layer.W_rec.shape == (4, 4)
    assert layer.state.shape == (4,)


def test_recurrent_step_output_shape():
    """step returns a vector of length n_neurons."""
    layer = SCRecurrentLayer(n_inputs=2, n_neurons=3, seed=1)
    out = layer.step(np.array([0.5, 0.2]))
    assert out.shape == (3,)


def test_recurrent_zero_input_zero_state():
    """Zero input with zero state yields zero output state."""
    layer = SCRecurrentLayer(n_inputs=2, n_neurons=2, seed=2)
    out = layer.step(np.zeros(2))
    assert np.allclose(out, 0.0)


def test_recurrent_state_bounds():
    """State values should be clipped into [0,1]."""
    layer = SCRecurrentLayer(n_inputs=1, n_neurons=2, seed=3)
    out = layer.step(np.array([10.0]))
    assert np.all(out >= 0.0)
    assert np.all(out <= 1.0)


def test_recurrent_reset_clears_state():
    """reset should restore zero state."""
    layer = SCRecurrentLayer(n_inputs=1, n_neurons=2, seed=4)
    _ = layer.step(np.array([0.9]))
    layer.reset()
    assert np.allclose(layer.state, 0.0)


def test_recurrent_seed_determinism():
    """Same seed yields identical weight matrices."""
    layer_a = SCRecurrentLayer(n_inputs=2, n_neurons=2, seed=10)
    layer_b = SCRecurrentLayer(n_inputs=2, n_neurons=2, seed=10)
    assert np.allclose(layer_a.W_in, layer_b.W_in)
    assert np.allclose(layer_a.W_rec, layer_b.W_rec)


def test_recurrent_input_strength_scales_weights():
    """Input weight magnitudes are bounded by input_strength."""
    layer = SCRecurrentLayer(n_inputs=2, n_neurons=2, input_strength=0.3, seed=11)
    assert np.max(layer.W_in) <= 0.3 + 1e-9


def test_recurrent_input_size_mismatch_raises():
    """Wrong input shape should raise a ValueError from numpy dot."""
    layer = SCRecurrentLayer(n_inputs=3, n_neurons=2, seed=12)
    with pytest.raises(ValueError):
        _ = layer.step(np.array([0.1, 0.2]))


def test_recurrent_state_updates_with_input():
    """Positive input should drive a non-zero state."""
    layer = SCRecurrentLayer(n_inputs=2, n_neurons=2, seed=13)
    out = layer.step(np.array([1.0, 1.0]))
    assert np.any(out > 0.0)


@pytest.mark.skipif(not _perf_enabled(), reason="Set SC_NEUROCORE_PERF=1 to enable perf checks.")
def test_recurrent_perf_small():
    """Benchmark a short recurrent step sequence."""
    layer = SCRecurrentLayer(n_inputs=4, n_neurons=8, seed=14)
    start = time.perf_counter()
    for _ in range(50):
        _ = layer.step(np.ones(4))
    elapsed = time.perf_counter() - start
    assert elapsed < 3.0
