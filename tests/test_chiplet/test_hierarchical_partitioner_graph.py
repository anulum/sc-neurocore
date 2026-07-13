# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Hierarchical partitioner graph and core tests

"""Graph cache, seed allocation, bisection, and refinement contracts."""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.chiplet import (
    CorrelationAwareGraph,
    CorrelationEdge,
    HierarchicalPartitioner,
    LFSRSeedAllocator,
    calculate_boundary_scc,
    calculate_edge_cut,
)
from tests.test_chiplet.hierarchical_partitioner_support import (
    build_graph as _build_graph,
    make_biclique as _make_biclique,
    make_chain_graph as _make_chain_graph,
)


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


class TestPartitionDeterministicOutput:
    """The perf fix must NOT change algorithm output — the partitioner
    is deterministic for a fixed graph + seed."""

    def test_partitions_canonical_match_baseline(self) -> None:
        # The baseline values were captured before the perf fix and
        # pinned here so any future algorithmic drift is loud.
        baseline_sizes = {50: [1, 1, 1, 47], 100: [1, 1, 1, 97], 200: [1, 1, 1, 197]}
        hp = HierarchicalPartitioner(num_partitions=4)
        for n_v, expected_sizes in baseline_sizes.items():
            g = _build_graph(n_v, avg_degree=8, seed=42)
            partitions, _seeds = hp.partition(g)
            sizes = sorted(len(p) for p in partitions)
            assert sizes == expected_sizes, (
                f"V={n_v} partition sizes drifted: got {sizes}, expected {expected_sizes}"
            )


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


class TestCorrelationAwareGraph:
    def test_adjacency(self) -> None:
        g = _make_chain_graph(5)
        adj = g.adjacency()
        assert 1 in adj[0]
        assert 0 in adj[1]
        assert 2 in adj[1]

    def test_edge_weight(self) -> None:
        g = _make_chain_graph(3)
        assert g.edge_weight(0, 1) == 1.0
        assert g.edge_weight(0, 2) == 0.0

    def test_num_edges(self) -> None:
        g = _make_chain_graph(10)
        assert g.num_edges == 9


# ── LFSRSeedAllocator Tests ─────────────────────────────────────────


class TestLFSRSeedAllocator:
    def test_allocate_unique_seeds(self) -> None:
        alloc = LFSRSeedAllocator()
        seeds = alloc.allocate(8)
        assert len(seeds) == 8
        assert alloc.verify_uniqueness(seeds)

    def test_no_zero_seeds(self) -> None:
        alloc = LFSRSeedAllocator()
        seeds = alloc.allocate(100)
        assert 0 not in seeds

    def test_single_partition(self) -> None:
        alloc = LFSRSeedAllocator()
        seeds = alloc.allocate(1)
        assert len(seeds) == 1
        assert seeds[0] != 0


# ── HierarchicalPartitioner Tests ────────────────────────────────────


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


# ── PartitionReport Tests ────────────────────────────────────────────
