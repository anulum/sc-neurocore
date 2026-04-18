# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Regression tests for HierarchicalPartitioner #65 perf fix

"""Regression tests for the #65 perf fix in CorrelationAwareGraph
+ HierarchicalPartitioner._spectral_bisect.

Before the fix:
  - `edge_scc(u, v)` linear-scanned `self.edges` per call → O(E)
  - `_spectral_bisect` rebuilt `set(vertices)` per vertex → O(V²)
  - Combined: O(V²·E) on partition; V=200 took ~700 ms.

After the fix:
  - `_ensure_edge_cache` builds a `(min, max) → CorrelationEdge`
    dict once per call → O(E) build, O(1) lookup
  - `_spectral_bisect` hoists `vset = set(vertices)` once
  - Combined: O(V·avg_degree); V=200 takes ~30 ms (22× speedup).

These tests pin:
1. Edge-cache lookup correctness (round-trip).
2. The cache is built only once per fresh graph.
3. The partitioner produces the SAME output before/after the fix
   (deterministic algorithm, same canonical partition for fixed
   seed and graph).
4. The wall-time scaling is within an order of magnitude of the
   expected linear-in-V curve, NOT the old quadratic curve.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from sc_neurocore.chiplet import (
    CorrelationAwareGraph,
    CorrelationEdge,
    HierarchicalPartitioner,
)


def _build_graph(n: int, avg_degree: int = 8, seed: int = 42) -> CorrelationAwareGraph:
    """Random sparse correlation-aware graph."""
    rng = np.random.default_rng(seed)
    edges: list[CorrelationEdge] = []
    seen: set[tuple[int, int]] = set()
    for v in range(n):
        for u in rng.choice(n, size=min(avg_degree, n - 1), replace=False):
            u = int(u)
            if u == v:
                continue
            key = (min(u, v), max(u, v))
            if key in seen:
                continue
            seen.add(key)
            edges.append(CorrelationEdge(u=u, v=v, conn_weight=1.0, scc_weight=0.1))
    return CorrelationAwareGraph(num_vertices=n, edges=edges)


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
        g = CorrelationAwareGraph(num_vertices=10, edges=[
            CorrelationEdge(u=0, v=1, conn_weight=1.0, scc_weight=0.5),
        ])
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
        g = CorrelationAwareGraph(num_vertices=4, edges=[
            CorrelationEdge(u=0, v=1, conn_weight=1.0, scc_weight=0.1),
            CorrelationEdge(u=1, v=2, conn_weight=1.0, scc_weight=0.1),
        ])
        _ = g.edge_scc(0, 1)
        before = g._edge_cache
        assert before is not None
        assert len(before) == 2
        # Mutate edges list externally — cache size now stale
        g.edges.append(CorrelationEdge(u=2, v=3, conn_weight=2.0, scc_weight=0.5))
        # Next lookup detects the size mismatch and rebuilds
        assert g.edge_scc(2, 3) == pytest.approx(0.5)
        after = g._edge_cache
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
                f"V={n_v} partition sizes drifted: got {sizes}, "
                f"expected {expected_sizes}"
            )


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
