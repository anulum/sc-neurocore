# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# Tests for sc_neurocore.topology (graph connectivity analysis)
from __future__ import annotations
import numpy as np
from sc_neurocore.topology import TopologyAnalyzer


class TestTopologyAnalyzer:
    def test_triangle(self):
        adj = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=np.float64)
        r = TopologyAnalyzer(adj).analyze()
        assert r.n_nodes == 3
        assert r.n_edges == 3
        assert r.clustering_coefficient == 1.0

    def test_chain(self):
        adj = np.zeros((4, 4))
        adj[0, 1] = adj[1, 0] = adj[1, 2] = adj[2, 1] = adj[2, 3] = adj[3, 2] = 1
        r = TopologyAnalyzer(adj).analyze()
        assert r.n_edges == 3
        assert r.clustering_coefficient == 0.0

    def test_disconnected(self):
        r = TopologyAnalyzer(np.zeros((5, 5))).analyze()
        assert r.n_edges == 0

    def test_complete(self):
        r = TopologyAnalyzer(np.ones((5, 5)) - np.eye(5)).analyze()
        assert r.density == 1.0

    def test_random_graph(self):
        rng = np.random.RandomState(42)
        adj = (rng.random((20, 20)) > 0.7).astype(float)
        np.fill_diagonal(adj, 0)
        adj = np.maximum(adj, adj.T)
        r = TopologyAnalyzer(adj).analyze()
        assert r.n_nodes == 20
        assert len(r.hub_neurons) <= 5

    def test_summary(self):
        adj = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
        assert "Topology" in TopologyAnalyzer(adj).analyze().summary()

    def test_from_weights(self):
        W = np.random.randn(10, 10) * 0.1
        W[np.abs(W) < 0.05] = 0
        assert TopologyAnalyzer(W).analyze().n_nodes == 10


class TestPathSampleCap:
    """n_path_samples constructor parameter (closes task #41)."""

    def test_default_cap_is_100(self):
        rng = np.random.RandomState(0)
        adj = (rng.random((10, 10)) < 0.3).astype(float)
        np.fill_diagonal(adj, 0)
        a = TopologyAnalyzer(adj)
        assert a.n_path_samples == 100

    def test_explicit_small_cap_does_not_crash(self):
        adj = (np.random.RandomState(3).random((200, 200)) < 0.05).astype(float)
        np.fill_diagonal(adj, 0)
        a = TopologyAnalyzer(adj, n_path_samples=5)
        rep = a.analyze()
        assert rep.n_nodes == 200
        # Sampled-from-5 path length is finite (>=0)
        assert rep.avg_path_length >= 0

    def test_cap_zero_falls_back_to_full_n(self):
        rng = np.random.RandomState(2)
        adj = (rng.random((40, 40)) < 0.2).astype(float)
        np.fill_diagonal(adj, 0)
        adj = np.maximum(adj, adj.T)
        a = TopologyAnalyzer(adj, n_path_samples=0)
        rep = a.analyze()
        assert rep.avg_path_length > 0

    def test_explicit_cap_attribute_persists(self):
        a = TopologyAnalyzer(np.zeros((5, 5)), n_path_samples=42)
        assert a.n_path_samples == 42
