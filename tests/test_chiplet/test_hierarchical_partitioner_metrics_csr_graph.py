# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCSRGraph from former test_hierarchical_partitioner_metrics.py

"""Focused suite: TestCSRGraph from former test_hierarchical_partitioner_metrics.py."""

from __future__ import annotations

from sc_neurocore.chiplet import (
    CSRGraph,
)
from tests.test_chiplet.hierarchical_partitioner_support import (
    make_chain_graph as _make_chain_graph,
)


class TestCSRGraph:
    """Construction and accessors of the CSR graph."""

    def test_from_edge_list(self) -> None:
        """A CSR graph built from a chain's edge list has the chain's vertex and edge counts."""
        g = _make_chain_graph(5)
        csr = CSRGraph.from_edge_list(5, g.edges)
        assert csr.num_vertices == 5
        assert csr.num_edges == 4

    def test_neighbors(self) -> None:
        """A middle vertex of a chain has both of its neighbours."""
        g = _make_chain_graph(5)
        csr = CSRGraph.from_edge_list(5, g.edges)
        n1 = csr.neighbors(1)
        assert 0 in n1
        assert 2 in n1

    def test_degree(self) -> None:
        """Chain endpoints have degree one and middle vertices degree two."""
        g = _make_chain_graph(5)
        csr = CSRGraph.from_edge_list(5, g.edges)
        assert csr.degree(0) == 1  # endpoint
        assert csr.degree(2) == 2  # middle

    def test_edge_weights(self) -> None:
        """The SCC weights of a vertex's edges keep their values."""
        g = _make_chain_graph(3, scc=0.5)
        csr = CSRGraph.from_edge_list(3, g.edges)
        scc_0 = csr.edge_scc(0)
        assert len(scc_0) == 1
        assert abs(scc_0[0] - 0.5) < 1e-6

    def test_vertex_weights(self) -> None:
        """Given vertex weights are kept and missing ones default to one."""
        g = _make_chain_graph(3)
        csr = CSRGraph.from_edge_list(3, g.edges, {0: 2.0, 1: 3.0})
        assert csr.vertex_weights[0] == 2.0
        assert csr.vertex_weights[1] == 3.0
        assert csr.vertex_weights[2] == 1.0  # default

    def test_to_csr(self) -> None:
        """A correlation-aware graph converts to CSR with its vertex and edge counts."""
        g = _make_chain_graph(10)
        csr = g.to_csr()
        assert csr.num_vertices == 10
        assert csr.num_edges == 9
