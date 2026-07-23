# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestEdgeCacheCorrectness from former test_hierarchical_partitioner_graph.py

"""Focused suite: TestEdgeCacheCorrectness from former test_hierarchical_partitioner_graph.py."""

from __future__ import annotations

from hierarchical_partitioner_graph_support import *  # noqa: F403

class TestEdgeCacheCorrectness:
    """The cached lookup must agree with a linear scan, on every edge
    AND on absent vertex pairs."""

    def test_edge_scc_matches_linear_scan(self) -> None:
        g = _build_graph(50, avg_degree=6, seed=7)
        for e in g.edges:
            # Symmetric lookup: both (u, v) and (v, u) return scc_weight.
            assert g.edge_scc(e.u, e.v) == pytest.approx(e.scc_weight)
            assert g.edge_scc(e.v, e.u) == pytest.approx(e.scc_weight)

    def test_edge_weight_matches_linear_scan(self) -> None:
        g = _build_graph(50, avg_degree=6, seed=7)
        for e in g.edges:
            assert g.edge_weight(e.u, e.v) == pytest.approx(e.conn_weight)
            assert g.edge_weight(e.v, e.u) == pytest.approx(e.conn_weight)

    def test_absent_pair_returns_zero(self) -> None:
        g = CorrelationAwareGraph(
            num_vertices=10,
            edges=[
                CorrelationEdge(u=0, v=1, conn_weight=1.0, scc_weight=0.5),
            ],
        )
        # (5, 6) is not an edge → both lookups must return 0.0
        assert g.edge_scc(5, 6) == 0.0
        assert g.edge_weight(5, 6) == 0.0
        # And the present edge still works
        assert g.edge_scc(0, 1) == 0.5
        assert g.edge_weight(0, 1) == 1.0
