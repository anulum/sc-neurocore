# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPathSampleCap from former test_graph_topology.py

"""Focused suite: TestPathSampleCap from former test_graph_topology.py."""

from __future__ import annotations

from tests.graph_topology_support import *  # noqa: F403

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
