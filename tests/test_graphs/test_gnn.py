"""Tests for StochasticGraphLayer message passing."""

import os
import time

import numpy as np
import pytest

from sc_neurocore.graphs.gnn import StochasticGraphLayer


def _perf_enabled() -> bool:
    return os.environ.get("SC_NEUROCORE_PERF") == "1"


def test_gnn_init_weights_shape():
    """Weights should be (n_features, n_features)."""
    np.random.seed(0)
    adj = np.eye(3)
    layer = StochasticGraphLayer(adj, n_features=4)
    assert layer.weights.shape == (4, 4)


def test_gnn_forward_shape():
    """Forward should return (N, n_features)."""
    adj = np.eye(2)
    layer = StochasticGraphLayer(adj, n_features=3)
    feats = np.ones((2, 3))
    out = layer.forward(feats)
    assert out.shape == (2, 3)


def test_gnn_zero_adj_returns_zero():
    """Zero adjacency should yield zero output."""
    adj = np.zeros((2, 2))
    layer = StochasticGraphLayer(adj, n_features=2)
    feats = np.random.random((2, 2))
    out = layer.forward(feats)
    assert np.allclose(out, 0.0)


def test_gnn_output_bounds():
    """Tanh output should be in [-1,1]."""
    adj = np.eye(3)
    layer = StochasticGraphLayer(adj, n_features=2)
    feats = np.random.random((3, 2))
    out = layer.forward(feats)
    assert np.all(out >= -1.0)
    assert np.all(out <= 1.0)


def test_gnn_isolated_node_safe():
    """Isolated nodes should not cause division by zero."""
    adj = np.array([[0, 0], [1, 0]], dtype=float)
    layer = StochasticGraphLayer(adj, n_features=2)
    feats = np.ones((2, 2))
    out = layer.forward(feats)
    assert np.all(np.isfinite(out))


def test_gnn_identity_adj_with_identity_weights():
    """Identity adjacency with identity weights should preserve features (tanh)."""
    adj = np.eye(2)
    layer = StochasticGraphLayer(adj, n_features=2)
    layer.weights = np.eye(2)
    feats = np.array([[0.1, 0.2], [0.3, 0.4]])
    out = layer.forward(feats)
    assert np.allclose(out, np.tanh(feats))


def test_gnn_determinism_with_seed():
    """Numpy seed should make weights deterministic."""
    np.random.seed(42)
    layer_a = StochasticGraphLayer(np.eye(2), n_features=2)
    np.random.seed(42)
    layer_b = StochasticGraphLayer(np.eye(2), n_features=2)
    assert np.allclose(layer_a.weights, layer_b.weights)


def test_gnn_input_shape_mismatch_raises():
    """Input shape mismatch should raise from numpy dot."""
    adj = np.eye(2)
    layer = StochasticGraphLayer(adj, n_features=2)
    feats = np.ones((3, 2))
    with pytest.raises(ValueError):
        _ = layer.forward(feats)


def test_gnn_degree_normalization_effect():
    """Degree normalization should scale aggregated features."""
    adj = np.array([[0, 1], [1, 1]], dtype=float)
    layer = StochasticGraphLayer(adj, n_features=1)
    layer.weights = np.ones((1, 1))
    feats = np.ones((2, 1))
    out = layer.forward(feats)
    assert out[1, 0] >= out[0, 0]


@pytest.mark.skipif(not _perf_enabled(), reason="Set SC_NEUROCORE_PERF=1 to enable perf checks.")
def test_gnn_perf_small():
    """Benchmark a small graph forward pass."""
    adj = np.eye(50)
    layer = StochasticGraphLayer(adj, n_features=8)
    feats = np.random.random((50, 8))
    start = time.perf_counter()
    _ = layer.forward(feats)
    elapsed = time.perf_counter() - start
    assert elapsed < 2.0
