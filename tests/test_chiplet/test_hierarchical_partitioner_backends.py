# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Hierarchical partitioner backend parity tests

"""Parity and performance contracts for maintained KL-refinement kernels."""

from __future__ import annotations

from collections.abc import Callable
import time

import pytest

from sc_neurocore.chiplet import HierarchicalPartitioner
from tests.test_chiplet.hierarchical_partitioner_support import build_graph as _build_graph


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
            f"Rust refine ({t_rs * 1000:.2f} ms) not >2× faster than "
            f"Python ({t_py * 1000:.2f} ms) at V=500 — perf regression?"
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
            ("rust", lambda: __import__("sc_neurocore_engine"), None),
            ("julia", lambda: __import__("juliacall"), None),
            ("go", lambda: None, "go_so"),
            ("mojo", lambda: None, "mojo_so"),
        ],
    )
    def test_dispatcher_backend_matches_python(
        self,
        backend: str,
        probe_fn: Callable[[], object],
        probe_arg: str | None,
    ) -> None:
        # Skip if the backend toolchain or built artefact is missing.
        try:
            probe_fn()
        except ImportError:
            pytest.skip(f"{backend}: prerequisite missing")
        if probe_arg in ("go_so", "mojo_so"):
            from pathlib import Path

            so = Path(__file__).resolve().parents[2] / (
                "src/sc_neurocore/accel/"
                + (
                    "go/partition/libpartition.so"
                    if probe_arg == "go_so"
                    else "mojo/partition/libpartition.so"
                )
            )
            if not so.is_file():
                pytest.skip(f"{backend}: {so.name} not built")

        g = _build_graph(100, avg_degree=8, seed=42)
        hp_py = HierarchicalPartitioner(num_partitions=4, kl_iterations=3, refine_backend="python")
        hp_x = HierarchicalPartitioner(num_partitions=4, kl_iterations=3, refine_backend=backend)
        parts_py, _ = hp_py.partition(g)
        parts_x, _ = hp_x.partition(g)
        pm_py = {v: i for i, p in enumerate(parts_py) for v in p}
        pm_x = {v: i for i, p in enumerate(parts_x) for v in p}
        assert pm_py == pm_x, (
            f"{backend} dispatcher disagrees with Python on "
            f"{sum(1 for v in pm_py if pm_py[v] != pm_x.get(v))} vertex assignments"
        )
