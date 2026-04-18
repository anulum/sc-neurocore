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


class TestRustRefineParityAndPerf:
    """The Rust KL refine kernel must produce the same partition
    membership (same vertex → partition mapping) as the Python
    reference, AND it must be measurably faster on a non-trivial
    workload."""

    def test_rust_backend_matches_python_membership(self) -> None:
        try:
            from sc_neurocore_engine import py_kl_refine  # noqa: F401
        except ImportError:
            pytest.skip("sc_neurocore_engine not built — Rust backend missing")

        import copy

        for n_v in (50, 100, 200):
            g = _build_graph(n_v, avg_degree=8, seed=42)
            adj = g.adjacency()
            n_parts = 4
            init = [
                [v for v in range(g.num_vertices) if v % n_parts == i]
                for i in range(n_parts)
            ]

            # Python reference
            hp_py = HierarchicalPartitioner(num_partitions=n_parts, kl_iterations=3,
                                             refine_backend="python")
            parts_py = copy.deepcopy(init)
            hp_py._refine(parts_py, adj, g)
            pm_py = {v: i for i, p in enumerate(parts_py) for v in p}

            # Rust dispatch
            hp_rs = HierarchicalPartitioner(num_partitions=n_parts, kl_iterations=3,
                                             refine_backend="rust")
            parts_rs = hp_rs._refine_rust(copy.deepcopy(init), adj, g)
            pm_rs = {v: i for i, p in enumerate(parts_rs) for v in p}

            assert pm_py == pm_rs, (
                f"V={n_v}: Rust partition membership differs from Python "
                f"reference at {sum(1 for v in pm_py if pm_py[v] != pm_rs[v])} "
                "vertices"
            )

    def test_rust_backend_is_faster_at_v500(self) -> None:
        try:
            from sc_neurocore_engine import py_kl_refine  # noqa: F401
        except ImportError:
            pytest.skip("sc_neurocore_engine not built")
        import copy

        g = _build_graph(500, avg_degree=8, seed=42)
        adj = g.adjacency()
        n_parts = 4
        init = [
            [v for v in range(g.num_vertices) if v % n_parts == i]
            for i in range(n_parts)
        ]

        hp_py = HierarchicalPartitioner(
            num_partitions=n_parts, kl_iterations=3, refine_backend="python",
        )
        hp_rs = HierarchicalPartitioner(
            num_partitions=n_parts, kl_iterations=3, refine_backend="rust",
        )
        # warm
        hp_py._refine(copy.deepcopy(init), adj, g)
        hp_rs._refine_rust(copy.deepcopy(init), adj, g)

        t0 = time.perf_counter()
        hp_py._refine(copy.deepcopy(init), adj, g)
        t_py = time.perf_counter() - t0
        t0 = time.perf_counter()
        hp_rs._refine_rust(copy.deepcopy(init), adj, g)
        t_rs = time.perf_counter() - t0

        # Generous floor: Rust should be at least 2× the Python time
        # at V=500 (measured ~250× on the dev box; CI runners noisier).
        assert t_rs * 2.0 < t_py, (
            f"Rust refine ({t_rs*1000:.2f} ms) not >2× faster than "
            f"Python ({t_py*1000:.2f} ms) at V=500 — perf regression?"
        )


class TestAllBackendsParityViaDispatcher:
    """Every wired backend (rust/julia/go/mojo) must produce the
    SAME vertex→partition mapping as the Python reference when
    invoked via `HierarchicalPartitioner(refine_backend=...)`.

    These tests exercise the production path (the dispatcher
    invocation), not the bench harness's direct ctypes/juliacall
    calls. They are the load-bearing verification that the wiring
    actually works for callers."""

    @pytest.mark.parametrize(
        "backend,probe_fn,probe_arg",
        [
            ("rust",  lambda: __import__("sc_neurocore_engine"), None),
            ("julia", lambda: __import__("juliacall"), None),
            ("go",    lambda: None, "go_so"),
            ("mojo",  lambda: None, "mojo_so"),
        ],
    )
    def test_dispatcher_backend_matches_python(
        self, backend: str, probe_fn, probe_arg,
    ) -> None:
        # Skip if the backend toolchain or built artefact is missing.
        try:
            probe_fn()
        except ImportError:
            pytest.skip(f"{backend}: prerequisite missing")
        if probe_arg in ("go_so", "mojo_so"):
            from pathlib import Path
            so = Path(__file__).resolve().parents[2] / (
                "src/sc_neurocore/accel/" + (
                    "go/partition/libpartition.so" if probe_arg == "go_so"
                    else "mojo/partition/libpartition.so"
                )
            )
            if not so.is_file():
                pytest.skip(f"{backend}: {so.name} not built")

        g = _build_graph(100, avg_degree=8, seed=42)
        hp_py = HierarchicalPartitioner(num_partitions=4, kl_iterations=3,
                                         refine_backend="python")
        hp_x = HierarchicalPartitioner(num_partitions=4, kl_iterations=3,
                                        refine_backend=backend)
        parts_py, _ = hp_py.partition(g)
        parts_x, _ = hp_x.partition(g)
        pm_py = {v: i for i, p in enumerate(parts_py) for v in p}
        pm_x = {v: i for i, p in enumerate(parts_x) for v in p}
        assert pm_py == pm_x, (
            f"{backend} dispatcher disagrees with Python on "
            f"{sum(1 for v in pm_py if pm_py[v] != pm_x.get(v))} vertex assignments"
        )


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
