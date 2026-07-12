#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Chiplet generator + hierarchical partitioner wall-clock benchmark

"""Reproducible benchmarks for the `sc_neurocore.chiplet` package.

Two suites:

1. **chiplet control** — `make_torus(rows, cols)`,
   `compute_decorrelation_seeds`, `estimate_package_energy`,
   `simulate_thermal`, and full `ChipletGenerator.generate()` at
   multiple die counts.
2. **hierarchical_partitioner** — `HierarchicalPartitioner.partition()`
   on multiple `(num_vertices, num_partitions)` cells.

Median + minimum over 30 repeats are reported per cell, together with source
digests, scheduler affinity, and host load.

**Backend policy:**

*chiplet_gen* ops (`make_torus`, `compute_decorrelation_seeds`,
`estimate_package_energy`, `simulate_thermal`) all run at 3 µs –
700 µs per call. FFI dispatch overhead (~1-5 µs for Rust PyO3,
~0.5-10 µs for Julia juliacall, ~1-3 µs for Go cgo+ctypes,
~1-3 µs for Mojo `mojo build --emit shared-lib` + ctypes) is
10-100 % of the op's total wall time for sub-ms kernels — a
native-language rewrite would at best halve that, often losing
the gain in marshalling. These ops are therefore documented as
EXEMPT rather than silently skipped.

*HierarchicalPartitioner.partition* is compute-heavy. Its KL-refinement
backends are measured separately by ``bench_kl_refine.py``.

Historical Rust, Julia, Go, and Mojo files named for ``chiplet_gen`` were
non-executable pasted-source or empty-return scaffolds. They are absent rather
than being reported as acceleration. The backend block below describes the
measured Python-only exemption for these package-control operations.

Usage:
    python benchmarks/bench_chiplet.py
    python benchmarks/bench_chiplet.py --json benchmarks/results/bench_chiplet.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import TypedDict

import numpy as np
import sc_neurocore.chiplet as chiplet_package

from sc_neurocore.chiplet import (
    ChipletDie,
    ChipletGenerator,
    ChipletTopology,
    CorrelationAwareGraph,
    CorrelationEdge,
    HierarchicalPartitioner,
    InterposerLink,
    InterposerTech,
    compute_decorrelation_seeds,
    estimate_package_energy,
    make_torus,
    simulate_thermal,
)


N_REPEATS = 30


class BackendStatus(TypedDict):
    """Availability and exemption evidence for one measured backend."""

    available: bool
    used: bool
    exemption: str | None


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of ``path``."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_hashes() -> dict[str, str]:
    """Bind results to the imported chiplet source files."""
    package_root = Path(chiplet_package.__file__).resolve().parent
    return {path.name: _sha256(path) for path in sorted(package_root.glob("*.py"))}


def _cpu_affinity() -> list[int] | None:
    """Return the Linux scheduler affinity when the platform exposes it."""
    try:
        return sorted(os.sched_getaffinity(0))
    except AttributeError:
        return None


def _bench(fn: Callable[[], object], repeats: int = N_REPEATS) -> tuple[float, float]:
    """Return (median_ms, min_ms) over `repeats` calls."""
    times_ms: list[float] = []
    fn()  # warm-up
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    times_ms.sort()
    return times_ms[len(times_ms) // 2], times_ms[0]


def _build_topology(n_dies: int) -> ChipletTopology:
    """Build a small mesh topology for benchmarking."""
    topo = ChipletTopology()
    for i in range(n_dies):
        topo.add_die(ChipletDie(die_id=i))
    # Link each die to the next (linear mesh)
    for i in range(n_dies - 1):
        topo.add_link(InterposerLink.from_tech(i, i + 1, InterposerTech.UCIE))
    return topo


def _build_correlation_graph(
    n_vertices: int,
    avg_degree: int = 8,
    seed: int = 42,
) -> CorrelationAwareGraph:
    """Build a correlation-aware graph with random sparse connectivity."""
    rng = np.random.default_rng(seed)
    edges: list[CorrelationEdge] = []
    seen: set[tuple[int, int]] = set()
    for v in range(n_vertices):
        neighbours = rng.choice(n_vertices, size=min(avg_degree, n_vertices - 1), replace=False)
        for u in neighbours:
            u = int(u)
            if u == v:
                continue
            key = (min(u, v), max(u, v))
            if key in seen:
                continue
            seen.add(key)
            edges.append(CorrelationEdge(u=u, v=v, conn_weight=1.0, scc_weight=0.1))
    return CorrelationAwareGraph(num_vertices=n_vertices, edges=edges)


# ─── chiplet_gen suites ───────────────────────────────────────────────


def bench_make_torus(rows: int, cols: int) -> tuple[float, float]:
    return _bench(lambda: make_torus(rows, cols))


def bench_compute_seeds(n_links: int) -> tuple[float, float]:
    """Build a topology with n_links and time the seed allocator."""
    topo = ChipletTopology()
    n_dies = n_links + 1
    for i in range(n_dies):
        topo.add_die(ChipletDie(die_id=i))
    for i in range(n_links):
        topo.add_link(InterposerLink.from_tech(i, i + 1, InterposerTech.UCIE))
    return _bench(lambda: compute_decorrelation_seeds(topo))


def bench_estimate_energy(n_dies: int) -> tuple[float, float]:
    topo = _build_topology(n_dies)
    return _bench(lambda: estimate_package_energy(topo, bits_per_link=1_000_000))


def bench_simulate_thermal(n_dies: int) -> tuple[float, float]:
    topo = _build_topology(n_dies)
    power_per_die = {die.die_id: 100.0 for die in topo.dies}  # 100 mW per die
    return _bench(lambda: simulate_thermal(topo, power_per_die))


def bench_generate(n_dies: int) -> tuple[float, float]:
    """Measure connected RTL and constraint generation for ``n_dies``."""
    topology = _build_topology(n_dies)
    generator = ChipletGenerator()
    return _bench(lambda: generator.generate(topology))


# ─── hierarchical_partitioner suite ───────────────────────────────────


def bench_partition(n_vertices: int, n_partitions: int) -> tuple[float, float]:
    g = _build_correlation_graph(n_vertices)
    hp = HierarchicalPartitioner(num_partitions=n_partitions)
    return _bench(lambda: hp.partition(g))


# ─── driver ───────────────────────────────────────────────────────────


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Chiplet wall-clock benchmark.")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    print("# Chiplet benchmark")
    print(f"# Repeats per cell: {N_REPEATS}")
    print(f"# Python: {platform.python_version()}, NumPy: {np.__version__}")
    print(f"# platform: {platform.platform()}")
    print(f"# load average: {os.getloadavg()}")
    print(f"# CPU affinity: {_cpu_affinity()}")
    print()

    print("## chiplet_gen")
    print(f"{'operation':<40}  {'median ms':>12}  {'min ms':>12}")
    print(f"{'-' * 40}  {'-' * 12}  {'-' * 12}")

    rows: list[dict[str, object]] = []

    # Note: make_torus signature is (rows, cols) — we sweep small grids.
    for label, (r, c) in [
        ("make_torus(2, 2)", (2, 2)),
        ("make_torus(4, 4)", (4, 4)),
        ("make_torus(8, 8)", (8, 8)),
    ]:
        med, mn = bench_make_torus(r, c)
        print(f"{label:<40}  {med:>12.3f}  {mn:>12.3f}")
        rows.append({"suite": "chiplet_gen", "op": label, "median_ms": med, "min_ms": mn})

    for n_links in [16, 64, 256]:
        med, mn = bench_compute_seeds(n_links)
        label = f"compute_decorrelation_seeds (n_links={n_links})"
        print(f"{label:<40}  {med:>12.3f}  {mn:>12.3f}")
        rows.append({"suite": "chiplet_gen", "op": label, "median_ms": med, "min_ms": mn})

    for n_dies in [4, 16, 64]:
        med, mn = bench_estimate_energy(n_dies)
        label = f"estimate_package_energy (n_dies={n_dies})"
        print(f"{label:<40}  {med:>12.3f}  {mn:>12.3f}")
        rows.append({"suite": "chiplet_gen", "op": label, "median_ms": med, "min_ms": mn})

    for n_dies in [4, 16, 64]:
        try:
            med, mn = bench_simulate_thermal(n_dies)
            label = f"simulate_thermal (n_dies={n_dies})"
            print(f"{label:<40}  {med:>12.3f}  {mn:>12.3f}")
            rows.append({"suite": "chiplet_gen", "op": label, "median_ms": med, "min_ms": mn})
        except Exception as exc:
            label = f"simulate_thermal (n_dies={n_dies})"
            print(f"{label:<40}  (skipped: {str(exc)[:30]})")
            rows.append({"suite": "chiplet_gen", "op": label, "skipped": str(exc)})

    for n_dies in [2, 8, 32]:
        med, mn = bench_generate(n_dies)
        label = f"ChipletGenerator.generate (n_dies={n_dies})"
        print(f"{label:<40}  {med:>12.3f}  {mn:>12.3f}")
        rows.append({"suite": "chiplet_gen", "op": label, "median_ms": med, "min_ms": mn})

    print()
    print("## hierarchical_partitioner")
    print("# KL-refinement backends are measured by bench_kl_refine.py.")
    print(f"{'operation':<40}  {'median ms':>12}  {'min ms':>12}")
    print(f"{'-' * 40}  {'-' * 12}  {'-' * 12}")
    for n_v, n_p in [(50, 2), (100, 4), (200, 4)]:
        try:
            med, mn = bench_partition(n_v, n_p)
            label = f"partition (V={n_v}, P={n_p})"
            print(f"{label:<40}  {med:>12.3f}  {mn:>12.3f}")
            rows.append({"suite": "partitioner", "op": label, "median_ms": med, "min_ms": mn})
        except Exception as exc:
            label = f"partition (V={n_v}, P={n_p})"
            print(f"{label:<40}  (skipped: {str(exc)[:30]})")
            rows.append({"suite": "partitioner", "op": label, "skipped": str(exc)})

    # Backend status for the chiplet package-control operations above.
    # The partitioner's real multi-language KL kernels are measured by
    # benchmarks/bench_kl_refine.py and are not represented by this block.
    backends_status: dict[str, BackendStatus] = {
        "python": {
            "available": True,
            "used": True,
            "exemption": None,
        },
        "rust": {
            "available": False,
            "used": False,
            "exemption": (
                "No installed chiplet-control Rust kernel: numerical analysis "
                "operations are sub-ms, while RTL generation is text and graph "
                "orchestration rather than a numerical kernel. The former "
                "rust/safety/chiplet_gen.rs file was nonfunctional scaffold."
            ),
        },
        "julia": {
            "available": False,
            "used": False,
            "exemption": (
                "No installed chiplet-control Julia kernel: numerical analysis "
                "is sub-ms and juliacall first-call JIT is about 5 s; RTL emission "
                "is textual orchestration. The former chiplet_gen.jl file was not valid Julia."
            ),
        },
        "go": {
            "available": False,
            "used": False,
            "exemption": (
                "No installed chiplet-control Go kernel: the numerical operations "
                "are sub-ms and RTL emission is textual orchestration. The former "
                "service contained empty functions."
            ),
        },
        "mojo": {
            "available": False,
            "used": False,
            "exemption": (
                "No installed chiplet-control Mojo kernel: the numerical operations "
                "are sub-ms and RTL emission is textual orchestration. The former "
                "kernel stored Python lines as strings and returned zero."
            ),
        },
    }

    print()
    print("# Chiplet-control backend status")
    for name, info in backends_status.items():
        tag = "USED" if info["used"] else "EXEMPT"
        print(f"  {name:<8} {tag:<8}  {info['exemption'] or '-'}")

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "n_repeats": N_REPEATS,
            "load_average": os.getloadavg(),
            "cpu_affinity": _cpu_affinity(),
            "benchmark_sha256": _sha256(Path(__file__)),
            "source_sha256": _source_hashes(),
            "rows": rows,
            "backends": backends_status,
        }
        args.json.write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
