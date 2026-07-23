# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestHierarchicalPartitioner from former test_hierarchical_partitioner_graph.py

"""Focused suite: TestHierarchicalPartitioner from former test_hierarchical_partitioner_graph.py."""

from __future__ import annotations

from hierarchical_partitioner_graph_support import *  # noqa: F403

class TestHierarchicalPartitioner:
    def test_single_partition(self) -> None:
        g = _make_chain_graph(10)
        hp = HierarchicalPartitioner(num_partitions=1)
        parts, seeds = hp.partition(g)
        assert len(parts) == 1
        assert len(parts[0]) == 10

    def test_two_partition_chain(self) -> None:
        g = _make_chain_graph(20)
        hp = HierarchicalPartitioner(num_partitions=2, seed=42)
        parts, seeds = hp.partition(g)
        assert len(parts) == 2
        assert sum(len(p) for p in parts) == 20
        assert len(seeds) == 2

    def test_four_partitions(self) -> None:
        g = _make_chain_graph(40)
        hp = HierarchicalPartitioner(num_partitions=4, seed=42)
        parts, seeds = hp.partition(g)
        assert len(parts) == 4
        assert sum(len(p) for p in parts) == 40
        assert len(seeds) == 4

    def test_biclique_preserves_vertices(self) -> None:
        g = _make_biclique(10, 10)
        hp = HierarchicalPartitioner(num_partitions=2, seed=42)
        parts, _ = hp.partition(g)
        all_vertices = sorted(v for p in parts for v in p)
        assert all_vertices == list(range(20))

    def test_tiny_graph(self) -> None:
        g = CorrelationAwareGraph(num_vertices=2, edges=[CorrelationEdge(0, 1)])
        hp = HierarchicalPartitioner(num_partitions=2)
        parts, seeds = hp.partition(g)
        assert sum(len(p) for p in parts) == 2

    def test_edge_cut_computed(self) -> None:
        g = _make_chain_graph(10)
        hp = HierarchicalPartitioner(num_partitions=2, seed=42)
        parts, _ = hp.partition(g)
        cut = calculate_edge_cut(g, parts)
        assert cut >= 1

    def test_boundary_scc(self) -> None:
        g = _make_chain_graph(10, scc=0.5)
        hp = HierarchicalPartitioner(num_partitions=2, seed=42)
        parts, _ = hp.partition(g)
        bscc = calculate_boundary_scc(g, parts)
        assert bscc >= 0.0

    def test_correlation_penalty_influences_cut(self) -> None:
        g = _make_chain_graph(20, scc=0.9)
        hp_no_penalty = HierarchicalPartitioner(num_partitions=2, correlation_penalty=0.0, seed=42)
        hp_penalty = HierarchicalPartitioner(num_partitions=2, correlation_penalty=5.0, seed=42)
        parts_np, _ = hp_no_penalty.partition(g)
        parts_p, _ = hp_penalty.partition(g)
        assert len(parts_np) == 2
        assert len(parts_p) == 2

    def test_seeds_unique_across_partitions(self) -> None:
        g = _make_chain_graph(100)
        hp = HierarchicalPartitioner(num_partitions=8, seed=42)
        parts, seeds = hp.partition(g)
        assert len(set(seeds)) == len(seeds)

    def test_large_graph(self) -> None:
        n = 500
        rng = np.random.default_rng(42)
        edges = []
        for i in range(n - 1):
            edges.append(CorrelationEdge(i, i + 1, 1.0, float(rng.random() * 0.3)))
        for _ in range(200):
            u, v = int(rng.integers(0, n)), int(rng.integers(0, n))
            if u != v:
                edges.append(CorrelationEdge(u, v, 1.0, float(rng.random() * 0.5)))
        g = CorrelationAwareGraph(num_vertices=n, edges=edges)
        hp = HierarchicalPartitioner(num_partitions=4, seed=42)
        parts, seeds = hp.partition(g)
        assert sum(len(p) for p in parts) == n
        assert len(seeds) == 4
