#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — QA Rust vs Python wall-clock benchmark (reproduces docs §6.1)

"""Reproducible benchmark for `SimulatedAnnealer.solve_ising`.

Measures the maintained ``engine/src/quantum.rs`` solver against the explicit
pure-Python fallback. Numbers vary by hardware; committed documentation must
refer to a source-hash-bound result rather than copying an old workstation
observation into a timeless speed claim.

**Run:**
    python benchmarks/bench_quantum_annealing_rust_vs_python.py
    python benchmarks/bench_quantum_annealing_rust_vs_python.py --json results/bench_qa.json
    python benchmarks/bench_quantum_annealing_rust_vs_python.py --sizes 10 30

Skips the Rust path with a clear message when the engine wheel is
not installed (`_HAS_RUST_QA = False`); the Python numbers still
print so the fallback can be characterised in isolation.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Literal

import numpy as np

from sc_neurocore.bridges import annealing_backends as backends
from sc_neurocore.bridges.quantum_annealing import (
    IsingModel,
    SimulatedAnnealer,
)


# Default protocol — kept identical to docs §6.1 to make the
# claim reproducible.
DEFAULT_SIZES = (20, 50, 100)
EDGE_PROBABILITY = 0.1
N_SWEEPS = 200
NUM_READS = 5
BETA_START = 0.1
BETA_END = 10.0
SEED = 42
_HAS_RUST_QA = backends.HAS_RUST_QA


def build_random_ising(n: int, p: float, seed: int) -> IsingModel:
    """Erdős–Rényi Ising with ±1 random h and J, fixed seed."""
    rng = np.random.default_rng(seed)
    h = {i: float(rng.choice([-1.0, 1.0])) for i in range(n)}
    j: dict[tuple[int, int], float] = {}
    for i in range(n):
        for k in range(i + 1, n):
            if rng.random() < p:
                j[(i, k)] = float(rng.choice([-1.0, 1.0]))
    return IsingModel(h=h, J=j, offset=0.0, n_qubits=n, source="bench_random")


def time_solver(
    model: IsingModel,
    backend: Literal["python", "rust"],
    n_repeats: int = 5,
) -> tuple[float, float, float]:
    """Run solve_ising `n_repeats` times, return (median_ms, min_ms, best_energy).

    Backend selection uses the bridge's explicit constructor contract, so the
    Python measurement cannot accidentally dispatch back into Rust.

    Median + min reported because Rust wall-times at small N are
    sub-millisecond and dominated by system noise; min is the
    closest estimate of the underlying compute cost, median is
    the typical-run figure.
    """
    if backend == "rust" and not _HAS_RUST_QA:
        raise RuntimeError("requested rust backend but _HAS_RUST_QA = False")
    sa = SimulatedAnnealer(
        n_sweeps=N_SWEEPS,
        beta_start=BETA_START,
        beta_end=BETA_END,
        seed=SEED,
        backend=backend,
    )

    # Warm-up: one short call so the timer excludes first-call overhead.
    sa.solve_ising(model, num_reads=1)
    times_ms: list[float] = []
    last_energy = 0.0
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        result = sa.solve_ising(model, num_reads=NUM_READS)
        times_ms.append((time.perf_counter() - t0) * 1000.0)
        last_energy = float(result["best_energy"])

    times_ms.sort()
    median_ms = times_ms[len(times_ms) // 2]
    min_ms = times_ms[0]
    return median_ms, min_ms, last_energy


def run(sizes: list[int]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    print(
        f"# QA Rust vs Python — sizes={sizes}, edge_p={EDGE_PROBABILITY}, "
        f"sweeps={N_SWEEPS}, reads={NUM_READS}, seed={SEED}"
    )
    print(f"# Python: {platform.python_version()}, NumPy: {np.__version__}")
    print(f"# _HAS_RUST_QA = {_HAS_RUST_QA}")
    print()
    print("# Each cell: median (min) over 5 repeats")
    print()
    print(
        f"{'N':>5}  {'py_med_ms':>12}  {'rust_med_ms':>14}  "
        f"{'speedup_med':>12}  {'py_E':>10}  {'rust_E':>10}"
    )
    print(f"{'-' * 5}  {'-' * 12}  {'-' * 14}  {'-' * 12}  {'-' * 10}  {'-' * 10}")

    for n in sizes:
        model = build_random_ising(n, EDGE_PROBABILITY, SEED)

        py_med, py_min, py_e = time_solver(model, "python", n_repeats=5)

        if _HAS_RUST_QA:
            rust_med, rust_min, rust_e = time_solver(model, "rust", n_repeats=5)
            speedup_med = py_med / rust_med if rust_med > 0 else float("inf")
            speedup_min = py_min / rust_min if rust_min > 0 else float("inf")
            rust_repr = f"{rust_med:>10.2f}({rust_min:.2f})"
            speedup_repr = f"{speedup_med:>12.1f}"
            rust_e_repr = f"{rust_e:>10.2f}"
        else:
            rust_med = float("nan")
            rust_min = float("nan")
            rust_e = float("nan")
            speedup_med = float("nan")
            speedup_min = float("nan")
            rust_repr = f"{'(skip)':>14}"
            speedup_repr = f"{'-':>12}"
            rust_e_repr = f"{'-':>10}"

        print(
            f"{n:>5d}  {py_med:>9.2f}({py_min:.1f})  {rust_repr}  "
            f"{speedup_repr}  {py_e:>10.2f}  {rust_e_repr}"
        )

        rows.append(
            {
                "n_qubits": n,
                "python_median_ms": py_med,
                "python_min_ms": py_min,
                "rust_median_ms": rust_med,
                "rust_min_ms": rust_min,
                "speedup_median": speedup_med,
                "speedup_min": speedup_min,
                "python_best_energy": py_e,
                "rust_best_energy": rust_e if _HAS_RUST_QA else None,
            }
        )

    return rows


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="QA Rust vs Python wall-clock benchmark (reproduces docs §6.1).",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_SIZES),
        help="Qubit counts to benchmark (default: 20 50 100).",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Optional path; if given, write results as JSON to this file.",
    )
    args = parser.parse_args(argv)

    rows = run(args.sizes)

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "_has_rust_qa": _HAS_RUST_QA,
            "protocol": {
                "edge_probability": EDGE_PROBABILITY,
                "n_sweeps": N_SWEEPS,
                "num_reads": NUM_READS,
                "beta_start": BETA_START,
                "beta_end": BETA_END,
                "seed": SEED,
            },
            "rows": rows,
        }
        args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
