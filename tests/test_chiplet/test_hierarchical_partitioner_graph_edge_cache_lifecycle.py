# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestEdgeCacheLifecycle from former test_hierarchical_partitioner_graph.py

"""Focused suite: TestEdgeCacheLifecycle from former test_hierarchical_partitioner_graph.py."""

from __future__ import annotations

from hierarchical_partitioner_graph_support import *  # noqa: F403

class TestEdgeCacheLifecycle:
    """The cache should be built once and reused, but rebuild after
    a manual edges-list mutation."""

    def test_cache_built_once(self) -> None:
        g = _build_graph(20, seed=3)
        # First call builds the cache
        _ = g.edge_scc(0, 1)
        cache1 = g._edge_cache
        assert cache1 is not None
        # Second call reuses it
        _ = g.edge_scc(2, 3)
        cache2 = g._edge_cache
        assert cache2 is cache1

    def test_cache_rebuilds_after_edge_append(self) -> None:
        # Use a clean 4-vertex graph with no duplicate or symmetric
        # edges so the cache size equals len(edges) by construction.
        g = CorrelationAwareGraph(
            num_vertices=4,
            edges=[
                CorrelationEdge(u=0, v=1, conn_weight=1.0, scc_weight=0.1),
                CorrelationEdge(u=1, v=2, conn_weight=1.0, scc_weight=0.1),
            ],
        )
        _ = g.edge_scc(0, 1)
        before = g._edge_cache
        assert before is not None
        assert len(before) == 2
        # Mutate edges list externally — cache size now stale
        g.edges.append(CorrelationEdge(u=2, v=3, conn_weight=2.0, scc_weight=0.5))
        # Next lookup detects the size mismatch and rebuilds
        assert g.edge_scc(2, 3) == pytest.approx(0.5)
        after = g._edge_cache
        assert after is not None
        assert after is not before
        assert len(after) == 3
