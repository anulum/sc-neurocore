# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestIncrementalRepartition from former test_hierarchical_partitioner_reporting.py

"""Focused suite: TestIncrementalRepartition from former test_hierarchical_partitioner_reporting.py."""

from __future__ import annotations

from hierarchical_partitioner_reporting_support import *  # noqa: F403

class TestIncrementalRepartition:
    def test_repartition_no_improvement(self) -> None:
        g = _make_chain_graph(4)
        parts = [[0, 1], [2, 3]]
        hp = HierarchicalPartitioner(num_partitions=2)
        new_parts, moves = hp.repartition_incremental(g, parts, max_moves=10)
        assert sum(len(p) for p in new_parts) == 4

    def test_repartition_improves_cut(self) -> None:
        g = _make_chain_graph(10, scc=0.5)
        parts = [[0, 2, 4, 6, 8], [1, 3, 5, 7, 9]]  # interleaved = bad
        hp = HierarchicalPartitioner(num_partitions=2, correlation_penalty=2.0)
        old_cut = calculate_edge_cut(g, parts)
        new_parts, moves = hp.repartition_incremental(g, parts, max_moves=20)
        new_cut = calculate_edge_cut(g, new_parts)
        assert new_cut <= old_cut  # should not get worse

    def test_zero_move_budget_returns_without_search(self) -> None:
        graph = _make_chain_graph(4)
        partitions = [[0, 1], [2, 3]]
        partitioner = HierarchicalPartitioner(num_partitions=2)
        result, moves = partitioner.repartition_incremental(
            graph,
            partitions,
            max_moves=0,
        )
        assert result == partitions
        assert moves == 0
