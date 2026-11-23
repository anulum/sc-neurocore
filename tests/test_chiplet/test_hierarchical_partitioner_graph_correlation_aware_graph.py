# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCorrelationAwareGraph from former test_hierarchical_partitioner_graph.py

"""Focused suite: TestCorrelationAwareGraph from former test_hierarchical_partitioner_graph.py."""

from __future__ import annotations

from tests.test_chiplet.hierarchical_partitioner_support import (
    make_chain_graph as _make_chain_graph,
)


class TestCorrelationAwareGraph:
    """Adjacency, edge weights and edge count of a correlation-aware graph."""

    def test_adjacency(self) -> None:
        """Chain neighbours appear in each other's adjacency lists."""
        g = _make_chain_graph(5)
        adj = g.adjacency()
        assert 1 in adj[0]
        assert 0 in adj[1]
        assert 2 in adj[1]

    def test_edge_weight(self) -> None:
        """A present edge has its weight and an absent pair weighs zero."""
        g = _make_chain_graph(3)
        assert g.edge_weight(0, 1) == 1.0
        assert g.edge_weight(0, 2) == 0.0

    def test_num_edges(self) -> None:
        """A chain of ten vertices has nine edges."""
        g = _make_chain_graph(10)
        assert g.num_edges == 9
