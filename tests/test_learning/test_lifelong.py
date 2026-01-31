"""Tests for EWC_SCLayer lifelong learning utilities."""

import os
import time

import numpy as np
import pytest

from sc_neurocore.learning.lifelong import EWC_SCLayer


def _perf_enabled() -> bool:
    return os.environ.get("SC_NEUROCORE_PERF") == "1"


def _make_layer(**overrides) -> EWC_SCLayer:
    params = dict(n_inputs=3, n_neurons=2, length=16, learning_rate=0.1, base_seed=1)
    params.update(overrides)
    return EWC_SCLayer(**params)


def test_ewc_shapes_initialized():
    """Fisher info and star weights should match (n_neurons, n_inputs)."""
    layer = _make_layer()
    assert layer.fisher_info.shape == (2, 3)
    assert layer.star_weights.shape == (2, 3)


def test_ewc_consolidate_copies_weights():
    """Consolidation should copy current weights."""
    layer = _make_layer()
    layer.consolidate_task()
    assert np.allclose(layer.star_weights, layer.get_weights())


def test_ewc_fisher_info_matches_weights():
    """Fisher info is set to current weights."""
    layer = _make_layer()
    layer.consolidate_task()
    assert np.allclose(layer.fisher_info, layer.get_weights())


def test_ewc_apply_penalty_no_error():
    """apply_ewc_penalty should be a no-op without errors."""
    layer = _make_layer()
    assert layer.apply_ewc_penalty() is None


def test_ewc_inherits_run_epoch():
    """EWC layer should execute run_epoch from base class."""
    layer = _make_layer(n_inputs=2, n_neurons=1, length=8)
    spikes = layer.run_epoch([0.2, 0.8])
    assert spikes.shape == (1, 8)


def test_ewc_zero_inputs_supported():
    """Zero inputs should still allow consolidation."""
    layer = _make_layer(n_inputs=0, n_neurons=1)
    layer.consolidate_task()
    assert layer.star_weights.shape == (1, 0)


def test_ewc_weight_bounds():
    """Weights should remain within bounds after consolidation."""
    layer = _make_layer(w_min=0.0, w_max=1.0)
    layer.consolidate_task()
    weights = layer.get_weights()
    assert np.all(weights >= 0.0)
    assert np.all(weights <= 1.0)


def test_ewc_deterministic_initial_weights():
    """Numpy seed should make initial weights deterministic."""
    np.random.seed(10)
    layer_a = _make_layer()
    np.random.seed(10)
    layer_b = _make_layer()
    assert np.allclose(layer_a.get_weights(), layer_b.get_weights())


def test_ewc_star_weights_update_on_consolidate():
    """Consolidation should update star weights to current values."""
    layer = _make_layer()
    layer.star_weights[:] = 0.0
    layer.consolidate_task()
    assert not np.allclose(layer.star_weights, 0.0)


@pytest.mark.skipif(not _perf_enabled(), reason="Set SC_NEUROCORE_PERF=1 to enable perf checks.")
def test_ewc_perf_small():
    """Benchmark a small consolidation workflow."""
    layer = _make_layer(n_inputs=4, n_neurons=4, length=32)
    start = time.perf_counter()
    _ = layer.run_epoch([0.1, 0.2, 0.3, 0.4])
    layer.consolidate_task()
    elapsed = time.perf_counter() - start
    assert elapsed < 3.0
