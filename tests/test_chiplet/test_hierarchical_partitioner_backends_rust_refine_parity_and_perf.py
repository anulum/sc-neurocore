# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestRustRefineParityAndPerf from former test_hierarchical_partitioner_backends.py

"""Focused suite: TestRustRefineParityAndPerf from former test_hierarchical_partitioner_backends.py."""

from __future__ import annotations

import sys
from pathlib import Path as _Path
from statistics import median

sys.path.insert(0, str(_Path(__file__).resolve().parent))
from hierarchical_partitioner_backends_support import *  # noqa: F403
from tests.performance_guard import assert_speedup_guard


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
            init = [[v for v in range(g.num_vertices) if v % n_parts == i] for i in range(n_parts)]

            # Python reference
            hp_py = HierarchicalPartitioner(
                num_partitions=n_parts, kl_iterations=3, refine_backend="python"
            )
            parts_py = copy.deepcopy(init)
            hp_py._refine(parts_py, adj, g)
            pm_py = {v: i for i, p in enumerate(parts_py) for v in p}

            # Rust dispatch
            hp_rs = HierarchicalPartitioner(
                num_partitions=n_parts, kl_iterations=3, refine_backend="rust"
            )
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
        init = [[v for v in range(g.num_vertices) if v % n_parts == i] for i in range(n_parts)]

        hp_py = HierarchicalPartitioner(
            num_partitions=n_parts,
            kl_iterations=3,
            refine_backend="python",
        )
        hp_rs = HierarchicalPartitioner(
            num_partitions=n_parts,
            kl_iterations=3,
            refine_backend="rust",
        )
        parts_py = copy.deepcopy(init)
        hp_py._refine(parts_py, adj, g)
        parts_rs = hp_rs._refine_rust(copy.deepcopy(init), adj, g)
        pm_py = {v: i for i, part in enumerate(parts_py) for v in part}
        pm_rs = {v: i for i, part in enumerate(parts_rs) for v in part}
        assert pm_py == pm_rs

        def timed_python() -> float:
            parts = copy.deepcopy(init)
            started = time.perf_counter()
            hp_py._refine(parts, adj, g)
            return time.perf_counter() - started

        def timed_rust() -> float:
            parts = copy.deepcopy(init)
            started = time.perf_counter()
            hp_rs._refine_rust(parts, adj, g)
            return time.perf_counter() - started

        python_samples = []
        rust_samples = []
        for sample_index in range(7):
            if sample_index % 2:
                rust_samples.append(timed_rust())
                python_samples.append(timed_python())
            else:
                python_samples.append(timed_python())
                rust_samples.append(timed_rust())

        assert_speedup_guard(
            label="Rust KL refine at V=500",
            baseline_seconds=median(python_samples),
            candidate_seconds=median(rust_samples),
            strict_minimum_speedup=2.0,
            smoke_minimum_speedup=0.25,
        )
