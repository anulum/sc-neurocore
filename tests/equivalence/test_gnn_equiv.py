"""Equivalence: v2 StochasticGraphLayer vs v3."""

import numpy as np
import pytest

from sc_neurocore.graphs.gnn import StochasticGraphLayer as V2GNN
from sc_neurocore_engine.graphs import StochasticGraphLayer as V3GNN


class TestGraphLayerEquivalence:
    @pytest.mark.parametrize(
        "n_nodes,n_features",
        [
            (5, 4),
            (10, 8),
            (20, 16),
        ],
    )
    def test_forward_matches_v2(self, n_nodes, n_features):
        rng = np.random.RandomState(42)
        adj = rng.randint(0, 2, (n_nodes, n_nodes)).astype(np.float64)
        adj = (adj + adj.T) / 2
        np.fill_diagonal(adj, 0.0)

        X = rng.uniform(0, 1, (n_nodes, n_features))

        np.random.seed(123)
        v2 = V2GNN(adj, n_features)
        v3 = V3GNN(adj, n_features)

        # Sync weights: v2 -> v3
        v3._engine.set_weights(v2.weights.flatten().tolist())

        v2_out = v2.forward(X)
        v3_out = v3.forward(X)

        np.testing.assert_allclose(v2_out, v3_out, atol=1e-12, err_msg="GNN output mismatch")

    def test_isolated_node(self):
        """Node with no edges should produce tanh(0) = 0."""
        adj = np.zeros((3, 3))
        adj[0, 1] = adj[1, 0] = 1.0
        # Node 2 is isolated
        X = np.ones((3, 4))

        v3 = V3GNN(adj, 4)
        out = v3.forward(X)
        np.testing.assert_allclose(out[2], 0.0, atol=1e-12)
