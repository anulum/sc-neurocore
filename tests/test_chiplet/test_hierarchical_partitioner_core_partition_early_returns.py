# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPartitionEarlyReturns from former test_hierarchical_partitioner_core.py

"""Focused suite: TestPartitionEarlyReturns from former test_hierarchical_partitioner_core.py."""

from __future__ import annotations

from sc_neurocore.chiplet import (
    HierarchicalPartitioner,
)
from tests.test_chiplet.hierarchical_partitioner_support import (
    build_graph as _build_graph,
)


class TestPartitionEarlyReturns:
    """Early-return paths of partitioning on tiny inputs.

    `partition()` and `_recursive_bisect` have early-return paths
    for tiny inputs that pytest --cov flagged uncovered.
    """

    def test_partition_with_fewer_vertices_than_partitions_pads(self) -> None:
        """With fewer vertices than partitions, each vertex gets its own part and the rest stay empty."""
        # n_v=3 < num_partitions=5 → return [[v] for v in vertices]
        # then pad with empty partitions.
        g = _build_graph(3, avg_degree=1, seed=11)
        hp = HierarchicalPartitioner(num_partitions=5)
        parts, seeds = hp.partition(g)
        assert len(parts) == 5
        assert sum(len(p) for p in parts) == 3
        # Two empty partitions
        assert sum(1 for p in parts if not p) == 2
        assert len(seeds) == 5

    def test_recursive_bisect_k_one_returns_input(self) -> None:
        """With one partition, partitioning returns every vertex in a single part."""
        hp = HierarchicalPartitioner(num_partitions=1)
        g = _build_graph(10, seed=2)
        parts, seeds = hp.partition(g)
        assert len(parts) == 1
        assert sorted(parts[0]) == list(range(10))

    def test_spectral_bisect_single_vertex(self) -> None:
        """Spectral bisection of a single vertex returns it with an empty other half."""
        # _spectral_bisect: `if len(vertices) <= 1: return vertices, []`
        hp = HierarchicalPartitioner(num_partitions=2)
        g = _build_graph(5, seed=2)
        adj = g.adjacency()
        a, b = hp._spectral_bisect([0], adj, g)
        assert a == [0] and b == []

    def test_recursive_bisect_direct_k_one(self) -> None:
        """Recursive bisection returns its input for k equal to one or for a single vertex."""
        # `_recursive_bisect(_, _, _, k=1)` is reachable internally
        # only via the recursion when k splits to 1; we exercise it
        # directly to cover the early-return branch.
        hp = HierarchicalPartitioner(num_partitions=4)
        g = _build_graph(8, seed=2)
        adj = g.adjacency()
        # k=1 → returns the input unchanged
        out = hp._recursive_bisect([0, 1, 2, 3], adj, g, k=1)
        assert out == [[0, 1, 2, 3]]
        # vertices length 1 → also early return
        out = hp._recursive_bisect([5], adj, g, k=2)
        assert out == [[5]]
