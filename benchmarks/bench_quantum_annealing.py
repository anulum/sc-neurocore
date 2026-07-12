#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""Benchmark: Python vs Rust quantum annealing hot paths.

Compares performance of:
1. Ising energy evaluation
2. Simulated annealing solver
3. Batch energy computation
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

from sc_neurocore.bridges.quantum_annealing import (
    IsingModel,
    SCToIsing,
    SimulatedAnnealer,
)

# Try Rust backend
try:
    import sc_neurocore_engine as _engine

    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False


def _build_model(n: int) -> tuple[IsingModel, NDArray[np.float64]]:
    """Build a random n-node SC network and compile to Ising."""
    rng = np.random.default_rng(42)
    adj = rng.random((n, n))
    adj = (adj + adj.T) / 2
    np.fill_diagonal(adj, 0)
    labels = [f"n{i}" for i in range(n)]
    model = SCToIsing().compile(adj, node_labels=labels)
    return model, adj


def _model_to_rust_args(
    model: IsingModel,
) -> dict[str, Any]:
    """Convert IsingModel to Rust function arguments."""
    h_indices = list(model.h.keys())
    h_values = list(model.h.values())
    j_i = [k[0] for k in model.J]
    j_j = [k[1] for k in model.J]
    j_values = list(model.J.values())
    return {
        "h_indices": h_indices,
        "h_values": h_values,
        "j_i": j_i,
        "j_j": j_j,
        "j_values": j_values,
    }


def bench_energy(sizes: list[int]) -> None:
    """Benchmark single Ising energy evaluation."""
    print("\n" + "=" * 60)
    print("  BENCHMARK: Ising Energy Evaluation")
    print("=" * 60)
    print(f"{'N':>6} {'Python (µs)':>14} {'Rust (µs)':>14} {'Speedup':>10}")
    print("-" * 60)

    for n in sizes:
        model, _ = _build_model(n)
        rng = np.random.default_rng(42)
        spins = {i: int(rng.choice((-1, 1))) for i in range(n)}

        # Python
        t0 = time.perf_counter()
        for _ in range(1000):
            model.energy(spins, backend="python")
        t_py = (time.perf_counter() - t0) / 1000 * 1e6

        # Rust
        if _HAS_RUST:
            args = _model_to_rust_args(model)
            spins_list = [spins.get(i, 1) for i in range(n)]
            t0 = time.perf_counter()
            for _ in range(1000):
                _engine.py_qa_ising_energy(
                    args["h_indices"],
                    args["h_values"],
                    args["j_i"],
                    args["j_j"],
                    args["j_values"],
                    spins_list,
                )
            t_rs = (time.perf_counter() - t0) / 1000 * 1e6
            speedup = t_py / t_rs
            print(f"{n:>6} {t_py:>14.1f} {t_rs:>14.1f} {speedup:>9.1f}x")
        else:
            print(f"{n:>6} {t_py:>14.1f} {'N/A':>14} {'N/A':>10}")


def bench_sa(sizes: list[int]) -> None:
    """Benchmark simulated annealing solver."""
    print("\n" + "=" * 60)
    print("  BENCHMARK: Simulated Annealing (1000 sweeps × 10 reads)")
    print("=" * 60)
    print(f"{'N':>6} {'Python (ms)':>14} {'Rust (ms)':>14} {'Speedup':>10}")
    print("-" * 60)

    n_sweeps = 1000
    num_reads = 10

    for n in sizes:
        model, _ = _build_model(n)

        # Python
        sa = SimulatedAnnealer(n_sweeps=n_sweeps, seed=42, backend="python")
        t0 = time.perf_counter()
        result_py = sa.solve_ising(model, num_reads=num_reads)
        t_py = (time.perf_counter() - t0) * 1e3

        # Rust
        if _HAS_RUST:
            args = _model_to_rust_args(model)
            t0 = time.perf_counter()
            result_rs = _engine.py_qa_simulated_annealing(
                args["h_indices"],
                args["h_values"],
                args["j_i"],
                args["j_j"],
                args["j_values"],
                n,
                model.offset,
                n_sweeps,
                num_reads,
                0.1,
                10.0,
                42,
            )
            t_rs = (time.perf_counter() - t0) * 1e3
            speedup = t_py / t_rs
            print(f"{n:>6} {t_py:>14.1f} {t_rs:>14.1f} {speedup:>9.1f}x")
            print(
                f"       E_py={result_py['best_energy']:.4f}  E_rs={result_rs['best_energy']:.4f}"
            )
        else:
            print(f"{n:>6} {t_py:>14.1f} {'N/A':>14} {'N/A':>10}")


def bench_batch_energy(sizes: list[int]) -> None:
    """Benchmark batch energy evaluation."""
    print("\n" + "=" * 60)
    print("  BENCHMARK: Batch Energy (10000 configurations)")
    print("=" * 60)
    print(f"{'N':>6} {'Python (ms)':>14} {'Rust (ms)':>14} {'Speedup':>10}")
    print("-" * 60)

    n_configs = 10000

    for n in sizes:
        model, _ = _build_model(n)
        rng = np.random.default_rng(42)
        configs_dict = [{i: int(rng.choice([-1, 1])) for i in range(n)} for _ in range(n_configs)]

        # Python
        t0 = time.perf_counter()
        energies_py = [model.energy(c) for c in configs_dict]
        t_py = (time.perf_counter() - t0) * 1e3

        # Rust
        if _HAS_RUST:
            args = _model_to_rust_args(model)
            configs_list = [[c.get(i, 1) for i in range(n)] for c in configs_dict]
            t0 = time.perf_counter()
            energies_rs = _engine.py_qa_batch_ising_energy(
                args["h_indices"],
                args["h_values"],
                args["j_i"],
                args["j_j"],
                args["j_values"],
                configs_list,
            )
            t_rs = (time.perf_counter() - t0) * 1e3
            speedup = t_py / t_rs
            print(f"{n:>6} {t_py:>14.1f} {t_rs:>14.1f} {speedup:>9.1f}x")
        else:
            print(f"{n:>6} {t_py:>14.1f} {'N/A':>14} {'N/A':>10}")


def main() -> None:
    """Run all benchmarks."""
    print("SC-NeuroCore Quantum Annealing Benchmark")
    print(f"Rust backend available: {_HAS_RUST}")
    print()

    sizes_small = [10, 20, 50, 100]
    sizes_sa = [10, 20, 50]

    bench_energy(sizes_small)
    bench_batch_energy([10, 20, 50])
    bench_sa(sizes_sa)

    print("\n" + "=" * 60)
    print("  BENCHMARK COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
