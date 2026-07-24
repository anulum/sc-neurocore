# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestModularity from former test_graph_topology.py

"""Focused suite: TestModularity from former test_graph_topology.py."""

from __future__ import annotations

from tests.graph_topology_support import *  # noqa: F403


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
