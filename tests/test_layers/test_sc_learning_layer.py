"""Tests for SCLearningLayer initialization, learning, and bounds."""

import os
import time

import numpy as np
import pytest

from sc_neurocore.layers.sc_learning_layer import SCLearningLayer


def _perf_enabled() -> bool:
    return os.environ.get("SC_NEUROCORE_PERF") == "1"


def _make_layer(**overrides) -> SCLearningLayer:
    params = dict(n_inputs=3, n_neurons=2, w_min=0.0, w_max=1.0, learning_rate=0.1, length=32, base_seed=7)
    params.update(overrides)
    return SCLearningLayer(**params)


def test_learning_layer_init_counts():
    """Neuron, synapse, and recorder counts should match sizes."""
    layer = _make_layer(n_inputs=4, n_neurons=3)
    assert len(layer.neurons) == 3
    assert len(layer.recorders) == 3
    assert len(layer.synapses) == 3
    assert len(layer.synapses[0]) == 4


def test_learning_layer_weights_shape():
    """get_weights returns (n_neurons, n_inputs)."""
    layer = _make_layer(n_inputs=5, n_neurons=2)
    weights = layer.get_weights()
    assert weights.shape == (2, 5)


def test_learning_layer_weights_within_bounds():
    """Initial weights lie within [w_min, w_max]."""
    layer = _make_layer(w_min=0.2, w_max=0.8)
    weights = layer.get_weights()
    assert np.all(weights >= 0.2)
    assert np.all(weights <= 0.8)


def test_learning_layer_run_epoch_shape():
    """run_epoch returns spike matrix with correct shape."""
    layer = _make_layer(n_inputs=2, n_neurons=3, length=16)
    spikes = layer.run_epoch([0.2, 0.8])
    assert spikes.shape == (3, 16)


def test_learning_layer_recorders_length_after_epoch():
    """Recorder length should equal epoch length after run."""
    layer = _make_layer(n_inputs=2, n_neurons=1, length=12)
    _ = layer.run_epoch([0.4, 0.6])
    assert len(layer.recorders[0].spikes) == 12


def test_learning_layer_learning_rate_zero_no_weight_change():
    """With learning_rate=0, weights should remain constant."""
    np.random.seed(0)
    layer = _make_layer(learning_rate=0.0)
    before = layer.get_weights().copy()
    _ = layer.run_epoch([0.5, 0.5, 0.5])
    after = layer.get_weights()
    assert np.allclose(before, after)


def test_learning_layer_seed_deterministic_initial_weights():
    """Setting numpy seed gives deterministic initial weights."""
    np.random.seed(123)
    layer_a = _make_layer()
    np.random.seed(123)
    layer_b = _make_layer()
    assert np.allclose(layer_a.get_weights(), layer_b.get_weights())


def test_learning_layer_zero_inputs_supported():
    """Zero-input layer should run without error and produce shape."""
    layer = _make_layer(n_inputs=0, n_neurons=1, length=8)
    spikes = layer.run_epoch([])
    assert spikes.shape == (1, 8)


def test_learning_layer_weights_float_dtype():
    """get_weights returns a floating array."""
    layer = _make_layer()
    weights = layer.get_weights()
    assert np.issubdtype(weights.dtype, np.floating)


@pytest.mark.skipif(not _perf_enabled(), reason="Set SC_NEUROCORE_PERF=1 to enable perf checks.")
def test_learning_layer_perf_small():
    """Benchmark a short learning epoch for performance sanity."""
    layer = _make_layer(n_inputs=4, n_neurons=4, length=64)
    start = time.perf_counter()
    _ = layer.run_epoch([0.2, 0.4, 0.6, 0.8])
    elapsed = time.perf_counter() - start
    assert elapsed < 3.0
