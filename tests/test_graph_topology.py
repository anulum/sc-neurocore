# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header

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


class TestModularity:
    """TopologyReport.modularity computation (closes task #42)."""

    def test_single_complete_graph_modularity_is_zero(self):
        """A single connected component has Q = 0 (no community structure)."""
        adj = np.ones((5, 5)) - np.eye(5)
        rep = TopologyAnalyzer(adj).analyze()
        assert abs(rep.modularity) < 1e-9

    def test_two_disjoint_cliques_modularity_is_half(self):
        """Two equal-size disjoint cliques give Q = 0.5 (Newman 2006 example)."""
        adj = np.zeros((10, 10))
        adj[:5, :5] = np.ones((5, 5)) - np.eye(5)
        adj[5:, 5:] = np.ones((5, 5)) - np.eye(5)
        rep = TopologyAnalyzer(adj).analyze()
        assert abs(rep.modularity - 0.5) < 1e-9

    def test_three_disjoint_cliques_higher_modularity(self):
        """Three equal cliques give Q approaching 1 - 1/k = 2/3 = 0.667."""
        adj = np.zeros((15, 15))
        for i in range(3):
            block = slice(5 * i, 5 * (i + 1))
            adj[block, block] = np.ones((5, 5)) - np.eye(5)
        rep = TopologyAnalyzer(adj).analyze()
        # Expected exactly 2/3 for 3 equal-size disjoint cliques
        assert abs(rep.modularity - 2 / 3) < 1e-9

    def test_empty_graph_modularity_zero(self):
        """Edgeless graph has Q = 0 (no edges to partition)."""
        rep = TopologyAnalyzer(np.zeros((5, 5))).analyze()
        assert rep.modularity == 0.0

    def test_modularity_explicit_partition(self):
        """Caller-supplied partition is honoured by _modularity()."""
        adj = np.zeros((6, 6))
        adj[:3, :3] = np.ones((3, 3)) - np.eye(3)
        adj[3:, 3:] = np.ones((3, 3)) - np.eye(3)
        a = TopologyAnalyzer(adj)
        # Correct partition: two communities
        q_correct = a._modularity(communities=[0, 0, 0, 1, 1, 1])
        # Wrong partition: everything in one community
        q_wrong = a._modularity(communities=[0, 0, 0, 0, 0, 0])
        # Wrong partition: each node alone
        q_singleton = a._modularity(communities=[0, 1, 2, 3, 4, 5])
        assert q_correct > q_wrong
        assert q_correct > q_singleton

    def test_modularity_partition_length_mismatch_raises(self):
        a = TopologyAnalyzer(np.zeros((5, 5)))
        import pytest

        with pytest.raises(ValueError, match="length"):
            a._modularity(communities=[0, 1, 2])  # length 3 != N=5
