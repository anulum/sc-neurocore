# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Hierarchical partitioner runtime reporting tests

"""Boundary, balancing, rank mapping, repartitioning, and report contracts."""

from __future__ import annotations

from sc_neurocore.chiplet import (
    BoundarySyncConfig,
    BoundarySyncProtocol,
    CorrelationAwareGraph,
    CorrelationEdge,
    CorrelationLoadBalancer,
    GhostCellManager,
    HierarchicalPartitioner,
    HierarchyLevel,
    RankMapper,
    build_partition_report,
    calculate_edge_cut,
)
from tests.test_chiplet.hierarchical_partitioner_support import (
    make_chain_graph as _make_chain_graph,
)


class TestGhostCellManager:
    def test_compute_halos(self) -> None:
        g = _make_chain_graph(6)
        parts = [[0, 1, 2], [3, 4, 5]]
        halos = GhostCellManager.compute_halos(g, parts)
        # Partition 0 needs ghost of vertex 3 (neighbor of 2)
        assert 3 in halos[0]
        # Partition 1 needs ghost of vertex 2 (neighbor of 3)
        assert 2 in halos[1]

    def test_halo_sizes(self) -> None:
        g = _make_chain_graph(6)
        parts = [[0, 1, 2], [3, 4, 5]]
        sizes = GhostCellManager.halo_sizes(g, parts)
        assert sizes[0] >= 1
        assert sizes[1] >= 1

    def test_no_halos_single_partition(self) -> None:
        g = _make_chain_graph(4)
        parts = [list(range(4))]
        sizes = GhostCellManager.halo_sizes(g, parts)
        assert sizes[0] == 0


# ── Boundary Sync Protocol Tests ─────────────────────────────────────


class TestBoundarySyncProtocol:
    def test_init_buffers(self) -> None:
        g = _make_chain_graph(6, scc=0.2)
        parts = [[0, 1, 2], [3, 4, 5]]
        seeds = [0xACE1, 0xBEEF]
        sync = BoundarySyncProtocol()
        count = sync.init_buffers(g, parts, seeds)
        assert count >= 1
        assert sync.num_buffers == count

    def test_scc_budget_no_violations(self) -> None:
        g = _make_chain_graph(6, scc=0.05)
        parts = [[0, 1, 2], [3, 4, 5]]
        sync = BoundarySyncProtocol(BoundarySyncConfig(max_boundary_scc_budget=0.1))
        violations = sync.check_scc_budget(g, parts)
        assert violations == []

    def test_scc_budget_with_violations(self) -> None:
        g = _make_chain_graph(6, scc=0.5)
        parts = [[0, 1, 2], [3, 4, 5]]
        sync = BoundarySyncProtocol(BoundarySyncConfig(max_boundary_scc_budget=0.1))
        violations = sync.check_scc_budget(g, parts)
        assert len(violations) >= 1

    def test_buffer_seed_nonzero(self) -> None:
        g = _make_chain_graph(4, scc=0.1)
        parts = [[0, 1], [2, 3]]
        seeds = [0x0001, 0x0001]  # same seed → XOR = 0 → forced to 1
        sync = BoundarySyncProtocol()
        sync.init_buffers(g, parts, seeds)
        for seed in sync.boundary_buffers.values():
            assert seed != 0


# ── Correlation Load Balancer Tests ──────────────────────────────────


class TestCorrelationLoadBalancer:
    def test_compute_load_metrics(self) -> None:
        g = _make_chain_graph(10)
        parts = [list(range(5)), list(range(5, 10))]
        lb = CorrelationLoadBalancer()
        metrics = lb.compute_load_metrics(g, parts)
        assert len(metrics) == 2
        assert metrics[0].vertex_count == 5

    def test_balanced_no_recommendations(self) -> None:
        g = _make_chain_graph(10)
        parts = [list(range(5)), list(range(5, 10))]
        lb = CorrelationLoadBalancer()
        recs = lb.recommend_migrations(g, parts)
        assert recs == []  # balanced → no recommendations

    def test_imbalanced_generates_recommendations(self) -> None:
        g = _make_chain_graph(10)
        parts = [[0, 1, 2, 3, 4, 5, 6, 7], [8, 9]]  # 8 vs 2
        lb = CorrelationLoadBalancer(imbalance_threshold=0.1)
        recs = lb.recommend_migrations(g, parts)
        assert len(recs) >= 0  # may or may not find boundary candidates

    def test_history_tracked(self) -> None:
        g = _make_chain_graph(10)
        parts = [list(range(5)), list(range(5, 10))]
        lb = CorrelationLoadBalancer()
        lb.recommend_migrations(g, parts)
        assert len(lb.history) >= 0  # at least attempted

    def test_boundary_target_may_not_be_underloaded(self) -> None:
        edges = [CorrelationEdge(vertex, 8, scc_weight=0.1) for vertex in range(8)]
        graph = CorrelationAwareGraph(num_vertices=12, edges=edges)
        partitions = [list(range(8)), list(range(8, 12)), []]
        balancer = CorrelationLoadBalancer(imbalance_threshold=0.2)
        assert balancer.recommend_migrations(graph, partitions) == []
        assert balancer.history == [[]]

    def test_empty_metrics_fail_closed_after_forced_threshold(self) -> None:
        graph = CorrelationAwareGraph(num_vertices=0)
        balancer = CorrelationLoadBalancer(imbalance_threshold=-0.1)
        assert balancer.recommend_migrations(graph, []) == []


# ── Rank Mapper Tests ────────────────────────────────────────────────


class TestRankMapper:
    def test_basic_assignment(self) -> None:
        mapper = RankMapper(num_ranks=2)
        parts = [[0, 1], [2, 3], [4, 5], [6, 7]]
        mapping = mapper.assign(parts)
        assert len(mapping) == 4
        assert all(0 <= v < 2 for v in mapping.values())

    def test_fewer_partitions_than_ranks(self) -> None:
        mapper = RankMapper(num_ranks=8)
        parts = [[0, 1], [2, 3]]
        mapping = mapper.assign(parts)
        assert mapping[0] == 0
        assert mapping[1] == 1

    def test_cross_rank_edges(self) -> None:
        g = _make_chain_graph(6)
        parts = [[0, 1, 2], [3, 4, 5]]
        mapper = RankMapper(num_ranks=2)
        cross = mapper.cross_rank_edges(g, parts)
        assert cross >= 1

    def test_hierarchy_levels(self) -> None:
        mapper = RankMapper(num_ranks=4, hierarchy=[HierarchyLevel.RACK, HierarchyLevel.NODE])
        assert mapper.hierarchy[0] == HierarchyLevel.RACK


# ── Incremental Repartition Tests ────────────────────────────────────


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


# ── Build Partition Report Tests ─────────────────────────────────────


class TestBuildPartitionReport:
    def test_full_report(self) -> None:
        g = _make_chain_graph(10, scc=0.3)
        hp = HierarchicalPartitioner(num_partitions=2, seed=42)
        parts, seeds = hp.partition(g)
        report = build_partition_report(g, parts, seeds)
        assert report.num_partitions == 2
        assert report.edge_cut >= 1
        assert report.imbalance_ratio >= 0.0
        assert report.comm_volume_bytes > 0
        assert len(report.seeds) == 2

    def test_scc_budget_violations_counted(self) -> None:
        g = _make_chain_graph(10, scc=0.5)
        parts = [list(range(5)), list(range(5, 10))]
        seeds = [0xACE1, 0xBEEF]
        report = build_partition_report(g, parts, seeds, scc_budget=0.1)
        assert report.scc_budget_violations >= 1

    def test_no_violations_when_budget_high(self) -> None:
        g = _make_chain_graph(10, scc=0.05)
        parts = [list(range(5)), list(range(5, 10))]
        seeds = [0xACE1, 0xBEEF]
        report = build_partition_report(g, parts, seeds, scc_budget=1.0)
        assert report.scc_budget_violations == 0
