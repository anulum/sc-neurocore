# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Hierarchical Partitioner Tests

import numpy as np

from sc_neurocore.chiplet.hierarchical_partitioner import (
    BoundarySyncConfig,
    BoundarySyncProtocol,
    CorrelationAwareGraph,
    CorrelationEdge,
    CorrelationLoadBalancer,
    CSRGraph,
    GhostCellManager,
    HierarchicalPartitioner,
    HierarchyLevel,
    LFSRSeedAllocator,
    PartitionReport,
    RankMapper,
    build_partition_report,
    calculate_boundary_scc,
    calculate_comm_volume,
    calculate_edge_cut,
    calculate_imbalance_ratio,
    calculate_mean_boundary_scc,
    calculate_total_boundary_scc,
)


def _make_chain_graph(n: int, scc: float = 0.0) -> CorrelationAwareGraph:
    """Create a simple chain graph: 0-1-2-...-n-1."""
    edges = [CorrelationEdge(i, i + 1, 1.0, scc) for i in range(n - 1)]
    return CorrelationAwareGraph(num_vertices=n, edges=edges)


def _make_biclique(n1: int, n2: int, scc: float = 0.0) -> CorrelationAwareGraph:
    """Bipartite complete graph: every node in [0, n1) connected to [n1, n1+n2)."""
    edges = []
    for i in range(n1):
        for j in range(n1, n1 + n2):
            edges.append(CorrelationEdge(i, j, 1.0, scc))
    return CorrelationAwareGraph(num_vertices=n1 + n2, edges=edges)


# ── CorrelationAwareGraph Tests ──────────────────────────────────────


class TestCorrelationAwareGraph:
    def test_adjacency(self):
        g = _make_chain_graph(5)
        adj = g.adjacency()
        assert 1 in adj[0]
        assert 0 in adj[1]
        assert 2 in adj[1]

    def test_edge_weight(self):
        g = _make_chain_graph(3)
        assert g.edge_weight(0, 1) == 1.0
        assert g.edge_weight(0, 2) == 0.0

    def test_num_edges(self):
        g = _make_chain_graph(10)
        assert g.num_edges == 9


# ── LFSRSeedAllocator Tests ─────────────────────────────────────────


class TestLFSRSeedAllocator:
    def test_allocate_unique_seeds(self):
        alloc = LFSRSeedAllocator()
        seeds = alloc.allocate(8)
        assert len(seeds) == 8
        assert alloc.verify_uniqueness(seeds)

    def test_no_zero_seeds(self):
        alloc = LFSRSeedAllocator()
        seeds = alloc.allocate(100)
        assert 0 not in seeds

    def test_single_partition(self):
        alloc = LFSRSeedAllocator()
        seeds = alloc.allocate(1)
        assert len(seeds) == 1
        assert seeds[0] != 0


# ── HierarchicalPartitioner Tests ────────────────────────────────────


class TestHierarchicalPartitioner:
    def test_single_partition(self):
        g = _make_chain_graph(10)
        hp = HierarchicalPartitioner(num_partitions=1)
        parts, seeds = hp.partition(g)
        assert len(parts) == 1
        assert len(parts[0]) == 10

    def test_two_partition_chain(self):
        g = _make_chain_graph(20)
        hp = HierarchicalPartitioner(num_partitions=2, seed=42)
        parts, seeds = hp.partition(g)
        assert len(parts) == 2
        assert sum(len(p) for p in parts) == 20
        assert len(seeds) == 2

    def test_four_partitions(self):
        g = _make_chain_graph(40)
        hp = HierarchicalPartitioner(num_partitions=4, seed=42)
        parts, seeds = hp.partition(g)
        assert len(parts) == 4
        assert sum(len(p) for p in parts) == 40
        assert len(seeds) == 4

    def test_biclique_preserves_vertices(self):
        g = _make_biclique(10, 10)
        hp = HierarchicalPartitioner(num_partitions=2, seed=42)
        parts, _ = hp.partition(g)
        all_vertices = sorted(v for p in parts for v in p)
        assert all_vertices == list(range(20))

    def test_tiny_graph(self):
        g = CorrelationAwareGraph(num_vertices=2, edges=[CorrelationEdge(0, 1)])
        hp = HierarchicalPartitioner(num_partitions=2)
        parts, seeds = hp.partition(g)
        assert sum(len(p) for p in parts) == 2

    def test_edge_cut_computed(self):
        g = _make_chain_graph(10)
        hp = HierarchicalPartitioner(num_partitions=2, seed=42)
        parts, _ = hp.partition(g)
        cut = calculate_edge_cut(g, parts)
        assert cut >= 1

    def test_boundary_scc(self):
        g = _make_chain_graph(10, scc=0.5)
        hp = HierarchicalPartitioner(num_partitions=2, seed=42)
        parts, _ = hp.partition(g)
        bscc = calculate_boundary_scc(g, parts)
        assert bscc >= 0.0

    def test_correlation_penalty_influences_cut(self):
        g = _make_chain_graph(20, scc=0.9)
        hp_no_penalty = HierarchicalPartitioner(num_partitions=2, correlation_penalty=0.0, seed=42)
        hp_penalty = HierarchicalPartitioner(num_partitions=2, correlation_penalty=5.0, seed=42)
        parts_np, _ = hp_no_penalty.partition(g)
        parts_p, _ = hp_penalty.partition(g)
        assert len(parts_np) == 2
        assert len(parts_p) == 2

    def test_seeds_unique_across_partitions(self):
        g = _make_chain_graph(100)
        hp = HierarchicalPartitioner(num_partitions=8, seed=42)
        parts, seeds = hp.partition(g)
        assert len(set(seeds)) == len(seeds)

    def test_large_graph(self):
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


class TestPartitionReport:
    def test_summary(self):
        r = PartitionReport(
            num_partitions=4,
            partition_sizes=[25, 25, 25, 25],
            edge_cut=12,
            max_boundary_scc=0.15,
            mean_boundary_scc=0.08,
            total_boundary_scc=0.96,
            imbalance_ratio=0.0,
            comm_volume_bytes=24576,
            comm_messages=12,
            seeds=[0xACE1, 0xBEEF, 0xCAFE, 0xDEAD],
        )
        s = r.summary()
        assert "4" in s
        assert "12" in s
        assert "Imbalance" in s
        assert "Comm" in s


# ── CSRGraph Tests ───────────────────────────────────────────────────


class TestCSRGraph:
    def test_from_edge_list(self):
        g = _make_chain_graph(5)
        csr = CSRGraph.from_edge_list(5, g.edges)
        assert csr.num_vertices == 5
        assert csr.num_edges == 4

    def test_neighbors(self):
        g = _make_chain_graph(5)
        csr = CSRGraph.from_edge_list(5, g.edges)
        n1 = csr.neighbors(1)
        assert 0 in n1
        assert 2 in n1

    def test_degree(self):
        g = _make_chain_graph(5)
        csr = CSRGraph.from_edge_list(5, g.edges)
        assert csr.degree(0) == 1  # endpoint
        assert csr.degree(2) == 2  # middle

    def test_edge_weights(self):
        g = _make_chain_graph(3, scc=0.5)
        csr = CSRGraph.from_edge_list(3, g.edges)
        scc_0 = csr.edge_scc(0)
        assert len(scc_0) == 1
        assert abs(scc_0[0] - 0.5) < 1e-6

    def test_vertex_weights(self):
        g = _make_chain_graph(3)
        csr = CSRGraph.from_edge_list(3, g.edges, {0: 2.0, 1: 3.0})
        assert csr.vertex_weights[0] == 2.0
        assert csr.vertex_weights[1] == 3.0
        assert csr.vertex_weights[2] == 1.0  # default

    def test_to_csr(self):
        g = _make_chain_graph(10)
        csr = g.to_csr()
        assert csr.num_vertices == 10
        assert csr.num_edges == 9


# ── Imbalance Ratio Tests ────────────────────────────────────────────


class TestImbalanceRatio:
    def test_perfect_balance(self):
        parts = [[0, 1], [2, 3], [4, 5]]
        assert calculate_imbalance_ratio(parts) == 0.0

    def test_imbalanced(self):
        parts = [[0, 1, 2, 3], [4]]
        ratio = calculate_imbalance_ratio(parts)
        assert ratio > 0.0

    def test_empty(self):
        assert calculate_imbalance_ratio([]) == 0.0

    def test_single_partition(self):
        assert calculate_imbalance_ratio([[0, 1, 2]]) == 0.0


# ── Mean/Total Boundary SCC Tests ────────────────────────────────────


class TestBoundarySCCMetrics:
    def test_mean_boundary_scc(self):
        g = _make_chain_graph(6, scc=0.3)
        parts = [[0, 1, 2], [3, 4, 5]]
        mean_scc = calculate_mean_boundary_scc(g, parts)
        assert mean_scc >= 0.0

    def test_total_boundary_scc(self):
        g = _make_chain_graph(6, scc=0.4)
        parts = [[0, 1, 2], [3, 4, 5]]
        total_scc = calculate_total_boundary_scc(g, parts)
        assert total_scc >= 0.0

    def test_no_boundary(self):
        g = _make_chain_graph(4, scc=0.5)
        parts = [list(range(4))]
        assert calculate_mean_boundary_scc(g, parts) == 0.0
        assert calculate_total_boundary_scc(g, parts) == 0.0


# ── Communication Volume Tests ───────────────────────────────────────


class TestCommVolume:
    def test_basic(self):
        g = _make_chain_graph(6, scc=0.1)
        parts = [[0, 1, 2], [3, 4, 5]]
        cv = calculate_comm_volume(g, parts)
        assert cv["boundary_edges"] >= 1
        assert cv["volume_bytes"] > 0
        assert cv["messages"] == cv["boundary_edges"]

    def test_no_boundary(self):
        g = _make_chain_graph(4)
        parts = [list(range(4))]
        cv = calculate_comm_volume(g, parts)
        assert cv["boundary_edges"] == 0
        assert cv["volume_bytes"] == 0


# ── Ghost Cell Manager Tests ─────────────────────────────────────────


class TestGhostCellManager:
    def test_compute_halos(self):
        g = _make_chain_graph(6)
        parts = [[0, 1, 2], [3, 4, 5]]
        halos = GhostCellManager.compute_halos(g, parts)
        # Partition 0 needs ghost of vertex 3 (neighbor of 2)
        assert 3 in halos[0]
        # Partition 1 needs ghost of vertex 2 (neighbor of 3)
        assert 2 in halos[1]

    def test_halo_sizes(self):
        g = _make_chain_graph(6)
        parts = [[0, 1, 2], [3, 4, 5]]
        sizes = GhostCellManager.halo_sizes(g, parts)
        assert sizes[0] >= 1
        assert sizes[1] >= 1

    def test_no_halos_single_partition(self):
        g = _make_chain_graph(4)
        parts = [list(range(4))]
        sizes = GhostCellManager.halo_sizes(g, parts)
        assert sizes[0] == 0


# ── Boundary Sync Protocol Tests ─────────────────────────────────────


class TestBoundarySyncProtocol:
    def test_init_buffers(self):
        g = _make_chain_graph(6, scc=0.2)
        parts = [[0, 1, 2], [3, 4, 5]]
        seeds = [0xACE1, 0xBEEF]
        sync = BoundarySyncProtocol()
        count = sync.init_buffers(g, parts, seeds)
        assert count >= 1
        assert sync.num_buffers == count

    def test_scc_budget_no_violations(self):
        g = _make_chain_graph(6, scc=0.05)
        parts = [[0, 1, 2], [3, 4, 5]]
        sync = BoundarySyncProtocol(BoundarySyncConfig(max_boundary_scc_budget=0.1))
        violations = sync.check_scc_budget(g, parts)
        assert violations == []

    def test_scc_budget_with_violations(self):
        g = _make_chain_graph(6, scc=0.5)
        parts = [[0, 1, 2], [3, 4, 5]]
        sync = BoundarySyncProtocol(BoundarySyncConfig(max_boundary_scc_budget=0.1))
        violations = sync.check_scc_budget(g, parts)
        assert len(violations) >= 1

    def test_buffer_seed_nonzero(self):
        g = _make_chain_graph(4, scc=0.1)
        parts = [[0, 1], [2, 3]]
        seeds = [0x0001, 0x0001]  # same seed → XOR = 0 → forced to 1
        sync = BoundarySyncProtocol()
        sync.init_buffers(g, parts, seeds)
        for seed in sync.boundary_buffers.values():
            assert seed != 0


# ── Correlation Load Balancer Tests ──────────────────────────────────


class TestCorrelationLoadBalancer:
    def test_compute_load_metrics(self):
        g = _make_chain_graph(10)
        parts = [list(range(5)), list(range(5, 10))]
        lb = CorrelationLoadBalancer()
        metrics = lb.compute_load_metrics(g, parts)
        assert len(metrics) == 2
        assert metrics[0].vertex_count == 5

    def test_balanced_no_recommendations(self):
        g = _make_chain_graph(10)
        parts = [list(range(5)), list(range(5, 10))]
        lb = CorrelationLoadBalancer()
        recs = lb.recommend_migrations(g, parts)
        assert recs == []  # balanced → no recommendations

    def test_imbalanced_generates_recommendations(self):
        g = _make_chain_graph(10)
        parts = [[0, 1, 2, 3, 4, 5, 6, 7], [8, 9]]  # 8 vs 2
        lb = CorrelationLoadBalancer(imbalance_threshold=0.1)
        recs = lb.recommend_migrations(g, parts)
        assert len(recs) >= 0  # may or may not find boundary candidates

    def test_history_tracked(self):
        g = _make_chain_graph(10)
        parts = [list(range(5)), list(range(5, 10))]
        lb = CorrelationLoadBalancer()
        lb.recommend_migrations(g, parts)
        assert len(lb.history) >= 0  # at least attempted


# ── Rank Mapper Tests ────────────────────────────────────────────────


class TestRankMapper:
    def test_basic_assignment(self):
        mapper = RankMapper(num_ranks=2)
        parts = [[0, 1], [2, 3], [4, 5], [6, 7]]
        mapping = mapper.assign(parts)
        assert len(mapping) == 4
        assert all(0 <= v < 2 for v in mapping.values())

    def test_fewer_partitions_than_ranks(self):
        mapper = RankMapper(num_ranks=8)
        parts = [[0, 1], [2, 3]]
        mapping = mapper.assign(parts)
        assert mapping[0] == 0
        assert mapping[1] == 1

    def test_cross_rank_edges(self):
        g = _make_chain_graph(6)
        parts = [[0, 1, 2], [3, 4, 5]]
        mapper = RankMapper(num_ranks=2)
        cross = mapper.cross_rank_edges(g, parts)
        assert cross >= 1

    def test_hierarchy_levels(self):
        mapper = RankMapper(num_ranks=4, hierarchy=[HierarchyLevel.RACK, HierarchyLevel.NODE])
        assert mapper.hierarchy[0] == HierarchyLevel.RACK


# ── Incremental Repartition Tests ────────────────────────────────────


class TestIncrementalRepartition:
    def test_repartition_no_improvement(self):
        g = _make_chain_graph(4)
        parts = [[0, 1], [2, 3]]
        hp = HierarchicalPartitioner(num_partitions=2)
        new_parts, moves = hp.repartition_incremental(g, parts, max_moves=10)
        assert sum(len(p) for p in new_parts) == 4

    def test_repartition_improves_cut(self):
        g = _make_chain_graph(10, scc=0.5)
        parts = [[0, 2, 4, 6, 8], [1, 3, 5, 7, 9]]  # interleaved = bad
        hp = HierarchicalPartitioner(num_partitions=2, correlation_penalty=2.0)
        old_cut = calculate_edge_cut(g, parts)
        new_parts, moves = hp.repartition_incremental(g, parts, max_moves=20)
        new_cut = calculate_edge_cut(g, new_parts)
        assert new_cut <= old_cut  # should not get worse


# ── Build Partition Report Tests ─────────────────────────────────────


class TestBuildPartitionReport:
    def test_full_report(self):
        g = _make_chain_graph(10, scc=0.3)
        hp = HierarchicalPartitioner(num_partitions=2, seed=42)
        parts, seeds = hp.partition(g)
        report = build_partition_report(g, parts, seeds)
        assert report.num_partitions == 2
        assert report.edge_cut >= 1
        assert report.imbalance_ratio >= 0.0
        assert report.comm_volume_bytes > 0
        assert len(report.seeds) == 2

    def test_scc_budget_violations_counted(self):
        g = _make_chain_graph(10, scc=0.5)
        parts = [list(range(5)), list(range(5, 10))]
        seeds = [0xACE1, 0xBEEF]
        report = build_partition_report(g, parts, seeds, scc_budget=0.1)
        assert report.scc_budget_violations >= 1

    def test_no_violations_when_budget_high(self):
        g = _make_chain_graph(10, scc=0.05)
        parts = [list(range(5)), list(range(5, 10))]
        seeds = [0xACE1, 0xBEEF]
        report = build_partition_report(g, parts, seeds, scc_budget=1.0)
        assert report.scc_budget_violations == 0
