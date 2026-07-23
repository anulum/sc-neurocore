# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestConnectomeGenerator from former test_utils_extended.py

"""Focused suite: TestConnectomeGenerator from former test_utils_extended.py."""

from __future__ import annotations

from tests.utils_extended_support import *  # noqa: F403

class TestConnectomeGenerator:
    def test_watts_strogatz_shape(self):
        adj = ConnectomeGenerator.generate_watts_strogatz(20, 4, 0.0)
        assert adj.shape == (20, 20)

    def test_watts_strogatz_no_self_loops(self):
        adj = ConnectomeGenerator.generate_watts_strogatz(20, 4, 0.0)
        assert np.all(np.diag(adj) == 0)

    def test_watts_strogatz_regular_ring(self):
        """With p_rewire=0, each node connects to k/2 forward neighbors."""
        np.random.seed(0)
        adj = ConnectomeGenerator.generate_watts_strogatz(10, 4, 0.0)
        # Each row should have at least 2 outgoing edges (k/2 = 2)
        row_sums = adj.sum(axis=1)
        assert np.all(row_sums >= 2)

    def test_watts_strogatz_full_rewire(self):
        """With p_rewire=1.0, graph is random but still connected."""
        np.random.seed(42)
        adj = ConnectomeGenerator.generate_watts_strogatz(10, 4, 1.0)
        # Should still have edges
        assert adj.sum() > 0
        # No self-loops
        assert np.all(np.diag(adj) == 0)

    def test_watts_strogatz_k_ge_n(self):
        """When k >= n, should return all-to-all minus diagonal."""
        adj = ConnectomeGenerator.generate_watts_strogatz(5, 5, 0.0)
        expected = np.ones((5, 5)) - np.eye(5)
        np.testing.assert_array_equal(adj, expected)

    def test_scale_free_shape(self):
        np.random.seed(0)
        adj = ConnectomeGenerator.generate_scale_free(20)
        assert adj.shape == (20, 20)

    def test_scale_free_edge_count(self):
        """Initial bidirectional edge (2 entries) + 13 directed edges = 15."""
        np.random.seed(0)
        adj = ConnectomeGenerator.generate_scale_free(15)
        # Initial: adj[0,1]=1 and adj[1,0]=1 (2 entries).
        # Nodes 2..14 each add 1 directed edge = 13 more.
        assert adj.sum() == 15

    def test_scale_free_no_self_loops(self):
        np.random.seed(0)
        adj = ConnectomeGenerator.generate_scale_free(15)
        assert np.all(np.diag(adj) == 0)
