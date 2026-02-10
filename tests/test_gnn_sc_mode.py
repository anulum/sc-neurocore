"""Tests for SC-mode GNN (new v3 capability)."""

import numpy as np

from sc_neurocore_engine.graphs import StochasticGraphLayer


class TestGNNScMode:
    def test_sc_mode_output_shape(self):
        adj = np.eye(5) + np.roll(np.eye(5), 1, axis=0)
        adj = (adj + adj.T).clip(0, 1)
        np.fill_diagonal(adj, 0.0)
        gnn = StochasticGraphLayer(adj, n_features=4)
        X = np.random.RandomState(42).uniform(0, 1, (5, 4))

        out = gnn.forward_sc(X, length=1024)
        assert out.shape == (5, 4)

    def test_sc_mode_deterministic(self):
        adj = np.eye(5)
        gnn = StochasticGraphLayer(adj, n_features=4)
        X = np.full((5, 4), 0.5)

        out1 = gnn.forward_sc(X, length=1024, seed=42)
        out2 = gnn.forward_sc(X, length=1024, seed=42)
        np.testing.assert_array_equal(out1, out2)

    def test_sc_mode_approximates_rate_mode(self):
        """Long bitstreams should converge toward rate-mode result."""
        rng = np.random.RandomState(42)
        adj = rng.randint(0, 2, (5, 5)).astype(np.float64)
        adj = (adj + adj.T) / 2
        np.fill_diagonal(adj, 0.0)

        gnn = StochasticGraphLayer(adj, n_features=4)
        X = rng.uniform(0.2, 0.8, (5, 4))

        out_rate = gnn.forward(X)
        out_sc = gnn.forward_sc(X, length=32768)

        np.testing.assert_allclose(
            out_rate,
            out_sc,
            atol=0.1,
            err_msg="SC mode should approximate rate mode with long bitstreams",
        )

    def test_isolated_node_sc(self):
        adj = np.zeros((3, 3))
        adj[0, 1] = adj[1, 0] = 1.0
        gnn = StochasticGraphLayer(adj, n_features=4)
        X = np.ones((3, 4)) * 0.5

        out = gnn.forward_sc(X, length=1024)
        np.testing.assert_allclose(out[2], 0.0, atol=0.05)
