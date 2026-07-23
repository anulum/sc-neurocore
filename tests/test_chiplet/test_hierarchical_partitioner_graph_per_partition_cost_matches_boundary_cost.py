# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPerPartitionCostMatchesBoundaryCost from former test_hierarchical_partitioner_graph.py

"""Focused suite: TestPerPartitionCostMatchesBoundaryCost from former test_hierarchical_partitioner_graph.py."""

from __future__ import annotations

from hierarchical_partitioner_graph_support import *  # noqa: F403

class TestPerPartitionCostMatchesBoundaryCost:
    """The new vector API `_per_partition_cost(v, P)` must agree
    with calling the legacy single-target `_boundary_cost(v, p)` for
    every p in 0..P. Otherwise the KL refine algorithm changes
    behaviour and the perf "win" is actually a regression."""

    def test_vector_matches_per_target_calls(self) -> None:
        # Build a small graph with realistic structure.
        g = _build_graph(20, avg_degree=5, seed=11)
        adj = g.adjacency()
        # Hand-construct a 3-partition split.
        partitions = [list(range(0, 7)), list(range(7, 14)), list(range(14, 20))]
        part_map: dict[int, int] = {}
        for i, part in enumerate(partitions):
            for v in part:
                part_map[v] = i
        hp = HierarchicalPartitioner(num_partitions=3)
        n_parts = len(partitions)
        for v in range(20):
            vec = hp._per_partition_cost(v, n_parts, part_map, adj, g)
            for p in range(n_parts):
                legacy = hp._boundary_cost(v, p, part_map, adj, g)
                assert vec[p] == pytest.approx(legacy, abs=1e-12), (
                    f"vector[{p}]={vec[p]} != legacy={legacy} for v={v}"
                )

    def test_unassigned_neighbour_is_excluded_from_partition_weight(self) -> None:
        graph = CorrelationAwareGraph(
            num_vertices=2,
            edges=[CorrelationEdge(0, 1, scc_weight=0.25)],
        )
        partitioner = HierarchicalPartitioner(
            num_partitions=1,
            correlation_penalty=2.0,
        )
        costs = partitioner._per_partition_cost(
            0,
            1,
            {},
            graph.adjacency(),
            graph,
        )
        assert costs == [1.5]
