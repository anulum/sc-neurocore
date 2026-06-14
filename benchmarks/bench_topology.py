#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Ollivier-Ricci curvature multi-language benchmark

"""Multi-language benchmark for ``ollivier_ricci_curvature``.

Times the exact optimal-transport curvature solve across the polyglot
backend chain (python / rust / julia / go / mojo) on weighted random
coupling graphs of increasing size, records the per-call wall-clock and
the parity gap against the NumPy reference, and writes a JSON artefact.

Usage::

    python benchmarks/bench_topology.py
    python benchmarks/bench_topology.py --json benchmarks/results/bench_topology.json

Measurement note: this harness is a functional / local-regression
benchmark. Wall-clock figures are recorded on a loaded developer
workstation and are explicitly **non-isolated** — do not promote the
speed numbers into release claims without an isolated-core rerun per
`BROADCAST_2026-06-04_benchmark_core_isolation`.
"""

from __future__ import annotations

import argparse
import json
import os as _os
import platform
import time
from pathlib import Path

import numpy as np

from sc_neurocore.math import topology
from sc_neurocore.math.topology import ollivier_ricci_curvature

N_REPEATS = 7


def _weighted_random(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    g = rng.random((n, n))
    g[g < 0.6] = 0.0
    g = 0.5 * (g + g.T)
    np.fill_diagonal(g, 0.0)
    return g


def _node_pairs(n: int) -> list[tuple[int, int]]:
    # A deterministic spread of pairs at varied graph distances.
    return [(0, n - 1), (1, n // 2), (2, n - 2), (3, n // 3), (n // 4, n - 3)]


def _probe_rust() -> tuple[bool, str]:
    if topology._HAS_RUST_TOPOLOGY:
        return True, ""
    return False, "sc_neurocore_engine.py_ollivier_ricci_curvature unavailable"


def _probe_julia() -> tuple[bool, str]:
    if topology._ensure_julia_loaded():
        return True, ""
    return False, "juliacall + accel/julia/math/topology.jl unavailable"


def _probe_go() -> tuple[bool, str]:
    if topology._ensure_go_loaded():
        return True, ""
    return False, (
        "accel/go/topology/libtopology.so not built — run "
        "`cd src/sc_neurocore/accel/go/topology && go build "
        "-buildmode=c-shared -o libtopology.so topology.go`"
    )


def _probe_mojo() -> tuple[bool, str]:
    if not _os.path.isfile(_os.path.expanduser("~/.pixi/bin/mojo")):
        return False, "mojo binary not at ~/.pixi/bin/mojo"
    if topology._ensure_mojo_loaded():
        return True, ""
    return False, (
        "accel/mojo/math/libtopology.so not built — run "
        "`cd src/sc_neurocore/accel/mojo/math && mojo build "
        "--emit shared-lib -o libtopology.so topology.mojo`"
    )


def _run_backend(
    graphs: list[np.ndarray], pairs: list[list[tuple[int, int]]], backend: str
) -> tuple[float, float, list[float]]:
    """Return (median_ms, min_ms, values) for one full sweep over all graphs."""

    def _sweep() -> list[float]:
        out: list[float] = []
        for graph, graph_pairs in zip(graphs, pairs):
            for i, j in graph_pairs:
                out.append(ollivier_ricci_curvature(graph, i, j, backend=backend))
        return out

    values = _sweep()  # warm-up (also JITs Julia)
    times_ms: list[float] = []
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        values = _sweep()
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    times_ms.sort()
    return times_ms[len(times_ms) // 2], times_ms[0], values


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Ollivier-Ricci curvature multi-language benchmark."
    )
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    sizes = [20, 50, 100]
    graphs = [_weighted_random(n, seed) for seed, n in enumerate(sizes, start=1)]
    pairs = [_node_pairs(n) for n in sizes]
    n_calls = sum(len(p) for p in pairs)

    print("# Ollivier-Ricci curvature benchmark")
    print(f"# Graphs: weighted random N={sizes}; {n_calls} curvature solves per sweep")
    print(f"# Repeats per backend: {N_REPEATS}")
    print(f"# Python: {platform.python_version()}, NumPy: {np.__version__}")
    print(f"# platform: {platform.platform()}")
    print("# isolation: non-isolated (loaded workstation) — functional/regression evidence")
    print()

    backends = {
        "python": (True, ""),
        "rust": _probe_rust(),
        "julia": _probe_julia(),
        "go": _probe_go(),
        "mojo": _probe_mojo(),
    }

    print(f"{'backend':<10}  {'available':<10}  reason / status")
    print(f"{'-' * 10}  {'-' * 10}  {'-' * 58}")
    for name, (avail, reason) in backends.items():
        print(f"{name:<10}  {'yes' if avail else 'no':<10}  {reason}")
    print()

    reference: list[float] | None = None
    rows: list[dict[str, object]] = []

    print(f"{'backend':<10}  {'median ms':>12}  {'min ms':>12}  {'parity Δ':>12}  {'speedup':>9}")
    print(f"{'-' * 10}  {'-' * 12}  {'-' * 12}  {'-' * 12}  {'-' * 9}")
    python_median: float | None = None
    for name, (avail, reason) in backends.items():
        if not avail:
            print(f"{name:<10}  {'(skip)':>12}  {'(skip)':>12}  {'-':>12}  {'-':>9}")
            rows.append({"backend": name, "skipped": True, "unavailable_reason": reason})
            continue
        median_ms, min_ms, values = _run_backend(graphs, pairs, name)
        if name == "python":
            reference = values
            python_median = median_ms
            parity = 0.0
        else:
            assert reference is not None
            parity = float(np.max(np.abs(np.asarray(values) - np.asarray(reference))))
        speedup = (python_median / median_ms) if python_median and median_ms > 0 else float("nan")
        print(f"{name:<10}  {median_ms:>12.3f}  {min_ms:>12.3f}  {parity:>12.2e}  {speedup:>8.2f}x")
        rows.append(
            {
                "backend": name,
                "median_ms": median_ms,
                "min_ms": min_ms,
                "parity_max_abs_diff": parity,
                "speedup_vs_python": speedup,
            }
        )

    report = {
        "benchmark": "ollivier_ricci_curvature",
        "workload": {"graph_sizes": sizes, "calls_per_sweep": n_calls, "repeats": N_REPEATS},
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "isolation": "non-isolated (loaded workstation)",
        },
        "results": rows,
    }

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"\nWrote {args.json}")
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))
