"""Tests for MyceliumLayer fungal computing dynamics."""

import os
import time

import numpy as np
import pytest

from sc_neurocore.exotic.fungal import MyceliumLayer


def _perf_enabled() -> bool:
    return os.environ.get("SC_NEUROCORE_PERF") == "1"


def test_fungal_conductance_shape_and_diag_zero():
    """Conductance should be NxN with zero diagonal."""
    np.random.seed(0)
    layer = MyceliumLayer(n_nodes=4)
    assert layer.conductance.shape == (4, 4)
    assert np.allclose(np.diag(layer.conductance), 0.0)


def test_fungal_step_output_shape():
    """step should return vector of length n_nodes."""
    layer = MyceliumLayer(n_nodes=3)
    out = layer.step(np.ones(3))
    assert out.shape == (3,)


def test_fungal_decay_with_zero_inputs():
    """Zero inputs should decay conductance."""
    np.random.seed(1)
    layer = MyceliumLayer(n_nodes=3, decay_rate=0.1, growth_rate=0.0)
    before = layer.conductance.copy()
    _ = layer.step(np.zeros(3))
    assert np.mean(layer.conductance) < np.mean(before)


def test_fungal_growth_with_inputs():
    """Positive inputs should increase conductance on average."""
    np.random.seed(2)
    layer = MyceliumLayer(n_nodes=3, decay_rate=0.0, growth_rate=0.2)
    before = layer.conductance.copy()
    _ = layer.step(np.ones(3))
    assert np.mean(layer.conductance) > np.mean(before)


def test_fungal_conductance_clipped():
    """Conductance values should be clipped to [0,1]."""
    layer = MyceliumLayer(n_nodes=3, growth_rate=5.0, decay_rate=0.0)
    _ = layer.step(np.ones(3))
    assert np.all(layer.conductance >= 0.0)
    assert np.all(layer.conductance <= 1.0)


def test_fungal_diagonal_remains_zero():
    """Diagonal should stay zero after updates."""
    layer = MyceliumLayer(n_nodes=3)
    _ = layer.step(np.ones(3))
    assert np.allclose(np.diag(layer.conductance), 0.0)


def test_fungal_flux_matches_known_conductance():
    """Flux should equal inputs dot conductance (calculated before update)."""
    layer = MyceliumLayer(n_nodes=2)
    conductance = np.array([[0.0, 0.5], [0.2, 0.0]])
    layer.conductance = conductance.copy()
    inputs = np.array([1.0, 2.0])
    # Calculate expected BEFORE calling step (since step updates conductance)
    expected = np.dot(inputs, conductance)  # [1*0 + 2*0.2, 1*0.5 + 2*0] = [0.4, 0.5]
    out = layer.step(inputs)
    assert np.allclose(out, expected)


def test_fungal_negative_inputs_safe():
    """Negative inputs should not break conductance bounds."""
    layer = MyceliumLayer(n_nodes=3)
    _ = layer.step(np.array([-1.0, -0.5, -0.2]))
    assert np.all(layer.conductance >= 0.0)


def test_fungal_seed_determinism():
    """Numpy seed should produce deterministic conductance."""
    np.random.seed(3)
    a = MyceliumLayer(n_nodes=3)
    np.random.seed(3)
    b = MyceliumLayer(n_nodes=3)
    assert np.allclose(a.conductance, b.conductance)


@pytest.mark.skipif(not _perf_enabled(), reason="Set SC_NEUROCORE_PERF=1 to enable perf checks.")
def test_fungal_perf_small():
    """Benchmark a small fungal step."""
    layer = MyceliumLayer(n_nodes=64)
    inputs = np.random.random(64)
    start = time.perf_counter()
    _ = layer.step(inputs)
    elapsed = time.perf_counter() - start
    assert elapsed < 2.0
