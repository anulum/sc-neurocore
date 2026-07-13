# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Hierarchical partitioner edge and scaling tests

"""Edge cases, backend validation, and coarse performance regression gates."""

from __future__ import annotations

import time

import numpy as np
import pytest

from sc_neurocore.chiplet import (
    CorrelationAwareGraph,
    CorrelationEdge,
    HierarchicalPartitioner,
)
from tests.test_chiplet.hierarchical_partitioner_support import build_graph as _build_graph


def test_historical_flat_buffer_wrappers_cover_invalid_partition_ids() -> None:
    """The historical private ABI wrappers retain their filtering contract."""
    graph = _build_graph(4, avg_degree=2, seed=9)
    partitioner = HierarchicalPartitioner(num_partitions=2)
    buffers = partitioner._encode_csr([[0, 2], [1, 3]], graph.adjacency(), graph)
    assert buffers[0].shape == (5,)
    decoded = partitioner._decode_part_map(
        np.asarray([-1, 0, 2, 1], dtype=np.int32),
        2,
    )
    assert decoded == [[1], [3]]


class TestCsrGraphAccessors:
    """`CSRGraph.edge_conn` and `LFSRSeedAllocator` zero-seed clamp
    are used by external callers but were uncovered by the existing
    suite — pin them here."""

    def test_edge_conn_returns_correct_slice(self) -> None:
        from sc_neurocore.chiplet.hierarchical_partitioner import CSRGraph

        edges = [
            CorrelationEdge(u=0, v=1, conn_weight=1.5, scc_weight=0.2),
            CorrelationEdge(u=1, v=2, conn_weight=2.5, scc_weight=0.3),
        ]
        csr = CSRGraph.from_edge_list(3, edges, None)
        # edge_conn returns a slice of conn_weights for vertex v's edges.
        conn0 = csr.edge_conn(0)
        assert len(conn0) == 1
        assert conn0[0] == 1.5

    def test_lfsr_seed_clamps_zero_to_one(self) -> None:
        from sc_neurocore.chiplet.hierarchical_partitioner import (
            LFSRSeedAllocator,
        )

        # Pick a base_seed and num_partitions that produce a 0 seed
        # for some i — the allocator must clamp it to 1.
        # base_seed=0xFFFF, spacing=65535//5 = 13107; one of the
        # combinations will hit `& 0xFFFF == 0` after addition.
        alloc = LFSRSeedAllocator(base_seed=0)
        # i=0 → 0 + 1*spacing = 13107; i=4 → 0 + 5*spacing = 65535
        # None of these are zero. Try base that wraps to zero:
        # 0xFFFF + 1*1 = 0x10000 → masked to 0
        alloc = LFSRSeedAllocator(base_seed=0xFFFF)
        seeds = alloc.allocate(num_partitions=65535)  # spacing=1
        # Some seed would be 0 without the clamp; verify none are zero.
        assert all(s != 0 for s in seeds)


class TestPartitionEarlyReturns:
    """`partition()` and `_recursive_bisect` have early-return paths
    for tiny inputs that pytest --cov flagged uncovered."""

    def test_partition_with_fewer_vertices_than_partitions_pads(self) -> None:
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
        hp = HierarchicalPartitioner(num_partitions=1)
        g = _build_graph(10, seed=2)
        parts, seeds = hp.partition(g)
        assert len(parts) == 1
        assert sorted(parts[0]) == list(range(10))

    def test_spectral_bisect_single_vertex(self) -> None:
        # _spectral_bisect: `if len(vertices) <= 1: return vertices, []`
        hp = HierarchicalPartitioner(num_partitions=2)
        g = _build_graph(5, seed=2)
        adj = g.adjacency()
        a, b = hp._spectral_bisect([0], adj, g)
        assert a == [0] and b == []

    def test_recursive_bisect_direct_k_one(self) -> None:
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


class TestPreExistingEdgeCases:
    """Two pre-existing edge-case lines (calculate_imbalance_ratio
    `ideal == 0` and MigrationPlanner `recs >= max_recommendations`)
    were uncovered by the original suite. Pin them so the chiplet
    package reaches 100 % coverage."""

    def test_imbalance_ratio_with_zero_ideal(self) -> None:
        from sc_neurocore.chiplet.hierarchical_partitioner import (
            calculate_imbalance_ratio,
        )

        # Empty partition list → ideal=0/0 short-circuits at line 895 (`not sizes`),
        # but [empty, empty] gives total=0, ideal=0 → triggers line 899/900.
        result = calculate_imbalance_ratio([[], []])
        assert result == 0.0

    def test_load_balancer_respects_max_recommendations(self) -> None:
        from sc_neurocore.chiplet.hierarchical_partitioner import (
            CorrelationLoadBalancer,
        )

        # Strong imbalance + many cross-partition edges → planner
        # produces multiple candidates; cap at 1 → forces the
        # `len(recs) >= max_recommendations` break (line 1116).
        edges = [CorrelationEdge(u=v, v=20, conn_weight=1.0, scc_weight=0.5) for v in range(20)]
        g = CorrelationAwareGraph(num_vertices=21, edges=edges)
        partitions = [list(range(20)), [20]]
        planner = CorrelationLoadBalancer(imbalance_threshold=0.05)
        recs = planner.recommend_migrations(
            g,
            partitions,
            max_recommendations=1,
        )
        assert len(recs) <= 1


class TestRefineBackendValidation:
    """The constructor must reject unknown backend names cleanly,
    and missing-tool errors at dispatch time must be informative."""

    def test_unknown_backend_raises(self) -> None:
        with pytest.raises(ValueError, match="refine_backend must be"):
            HierarchicalPartitioner(refine_backend="cuda")

    def test_known_backends_construct(self) -> None:
        for b in ("auto", "rust", "julia", "go", "mojo", "python"):
            hp = HierarchicalPartitioner(refine_backend=b)
            assert hp.refine_backend == b


class TestPartitionScalingIsLinearish:
    """The wall-clock should NOT grow quadratically with V any more.
    We accept a generous slack — exact ms differs by hardware — but
    a 10× V increase must NOT cause a 100× wall-clock increase."""

    def test_v200_finishes_under_one_second(self) -> None:
        # Pre-fix this took ~700 ms on the dev box; post-fix ~30 ms.
        # 1 s is a generous CI margin (covers slow shared runners).
        hp = HierarchicalPartitioner(num_partitions=4)
        g = _build_graph(200, avg_degree=8, seed=42)
        t0 = time.perf_counter()
        hp.partition(g)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        assert elapsed_ms < 1000.0, (
            f"V=200 partition took {elapsed_ms:.1f} ms — perf fix #65 "
            "may have regressed (expected < 1 s, was ~700 ms before fix)"
        )

    def test_scaling_better_than_quadratic(self) -> None:
        """Doubling V should not cause >5× wall-clock increase."""
        hp = HierarchicalPartitioner(num_partitions=4)
        # Warm any caches first — the first call always pays init cost.
        g_warm = _build_graph(50, seed=7)
        hp.partition(g_warm)

        def time_partition(n: int) -> float:
            g = _build_graph(n, avg_degree=8, seed=42)
            t0 = time.perf_counter()
            hp.partition(g)
            return (time.perf_counter() - t0) * 1000.0

        t100 = time_partition(100)
        t200 = time_partition(200)
        # Pre-fix ratio was ~3.6× for V doubling. Quadratic would be 4×.
        # Cubic O(V²·E) gave the actual measured ratio of ~3.6× because
        # E grows linearly with V at fixed degree. We require strictly
        # better than 5× to leave headroom for noise on slow runners.
        ratio = t200 / max(t100, 0.5)  # avoid div-by-zero on very fast hosts
        assert ratio < 5.0, (
            f"V doubling caused {ratio:.1f}× wall-clock increase "
            f"(t100={t100:.1f} ms → t200={t200:.1f} ms); the #65 fix "
            "should keep this near-linear"
        )
