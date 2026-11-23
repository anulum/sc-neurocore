# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestTopologyAnalyzer from former test_graph_topology.py

"""Focused suite: TestTopologyAnalyzer from former test_graph_topology.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.topology import (
    TopologyAnalyzer,
)


class TestTopologyAnalyzer:
    """Analysing network topology."""

    def test_triangle(self) -> None:
        """A triangle has three edges and full clustering."""
        adj = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=np.float64)
        r = TopologyAnalyzer(adj).analyze()
        assert r.n_nodes == 3
        assert r.n_edges == 3
        assert r.clustering_coefficient == 1.0

    def test_chain(self) -> None:
        """A chain has three edges and no clustering."""
        adj = np.zeros((4, 4))
        adj[0, 1] = adj[1, 0] = adj[1, 2] = adj[2, 1] = adj[2, 3] = adj[3, 2] = 1
        r = TopologyAnalyzer(adj).analyze()
        assert r.n_edges == 3
        assert r.clustering_coefficient == 0.0

    def test_disconnected(self) -> None:
        """A disconnected graph has no edges."""
        r = TopologyAnalyzer(np.zeros((5, 5))).analyze()
        assert r.n_edges == 0

    def test_complete(self) -> None:
        """A complete graph has density 1."""
        r = TopologyAnalyzer(np.ones((5, 5)) - np.eye(5)).analyze()
        assert r.density == 1.0

    def test_random_graph(self) -> None:
        """A random graph lists at most five hubs."""
        rng = np.random.RandomState(42)
        adj = (rng.random((20, 20)) > 0.7).astype(float)
        np.fill_diagonal(adj, 0)
        adj = np.maximum(adj, adj.T)
        r = TopologyAnalyzer(adj).analyze()
        assert r.n_nodes == 20
        assert len(r.hub_neurons) <= 5

    def test_summary(self) -> None:
        """The summary names the topology."""
        adj = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
        assert "Topology" in TopologyAnalyzer(adj).analyze().summary()

    def test_from_weights(self) -> None:
        """A weight matrix is analysed as a graph."""
        W = np.random.randn(10, 10) * 0.1
        W[np.abs(W) < 0.05] = 0
        assert TopologyAnalyzer(W).analyze().n_nodes == 10
