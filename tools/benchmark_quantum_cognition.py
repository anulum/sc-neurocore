# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Automated Quantum Cognition Benchmark Suite

"""Run all quantum cognition benchmarks and produce structured results.

Usage::

    python tools/benchmark_quantum_cognition.py          # full suite
    python tools/benchmark_quantum_cognition.py --quick  # reduced iterations
    python tools/benchmark_quantum_cognition.py --json   # machine-parseable

Benchmarks all 4 backends (Python, Rust, Mojo, Julia) across:
    - SpinPoolMPS.apply_measurement
    - Neuron batch stepping
    - Radical pair singlet yield
    - Kane coupling matrix
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# Add source to path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from sc_neurocore.quantum_cognition.spin_pool import SpinPoolMPS
from sc_neurocore.quantum_cognition.fisher_posner import HybridFisherPosnerLIF
from sc_neurocore.quantum_cognition.radical_pair import RadicalPairModel
from sc_neurocore.quantum_cognition.kane_mapper import KaneSiliconMapper

_QC_DIR = _ROOT / "src" / "sc_neurocore" / "quantum_cognition"

# ─── Python Benchmarks ───

def bench_py_apply_measurement(n_sites: int, n_calls: int) -> float:
    pool = SpinPoolMPS(n_sites=n_sites, bond_dim=16)
    t0 = time.perf_counter()
    for i in range(n_calls):
        pool.apply_measurement(i % n_sites, 1.0)
    return (time.perf_counter() - t0) / n_calls * 1e6  # µs/call

def bench_py_neuron_step(n_neurons: int, n_steps: int) -> float:
    pool = SpinPoolMPS(n_sites=n_neurons, bond_dim=16)
    neurons = [HybridFisherPosnerLIF(i, pool) for i in range(n_neurons)]
    rng = np.random.default_rng(42)
    currents = rng.normal(25.0, 5.0, size=n_neurons)
    t0 = time.perf_counter()
    for _ in range(n_steps):
        for j, n in enumerate(neurons):
            n.step(currents[j])
    dt = time.perf_counter() - t0
    return dt / (n_steps * n_neurons) * 1e6  # µs/neuron-step

def bench_py_singlet_yield(n_calls: int) -> float:
    model = RadicalPairModel()
    fields = np.linspace(0, 1e-3, n_calls)
    t0 = time.perf_counter()
    for b in fields:
        model.singlet_yield(b)
    return (time.perf_counter() - t0) / n_calls * 1e6

def bench_py_coupling_matrix(n_sites: int) -> float:
    mapper = KaneSiliconMapper(spacing_nm=10.0, topology="linear")
    t0 = time.perf_counter()
    mapper.map_pool_to_register(n_sites)
    return (time.perf_counter() - t0) * 1e3  # ms

# ─── Rust Benchmarks ───

def _run_rust_benchmark(rs_file: str, opt: str = "-C opt-level=2") -> dict[str, Any] | None:
    rs_path = _QC_DIR / rs_file
    if not rs_path.exists():
        return None
    bin_path = f"/tmp/{rs_file.replace('.rs', '_bench')}"
    result = subprocess.run(
        ["rustc", str(rs_path), "-o", bin_path] + opt.split(),
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        return None
    result = subprocess.run(
        [bin_path], capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        return None
    return {"output": result.stdout}

def bench_e2e_pipeline(n_neurons: int, n_chunks: int) -> dict[str, float]:
    """Measure end-to-end neuron stepping throughput.

    Bypasses PennyLane optimize_phases to measure pure
    classical neural stepping + quantum spin pool throughput.
    """
    from sc_neurocore.quantum_cognition.spin_pool import SpinPoolMPS
    from sc_neurocore.quantum_cognition.fisher_posner import HybridFisherPosnerLIF

    pool = SpinPoolMPS(n_sites=n_neurons)
    neurons = [HybridFisherPosnerLIF(i, pool) for i in range(n_neurons)]
    rng = np.random.default_rng(42)

    total_spikes = 0
    t0 = time.perf_counter()
    for chunk_i in range(n_chunks):
        currents = rng.normal(25.0, 10.0, size=n_neurons) * 2.0
        for j, neuron in enumerate(neurons):
            _, spiked = neuron.step(float(currents[j]))
            if spiked:
                total_spikes += 1
    dt = time.perf_counter() - t0

    return {
        "total_ms": dt * 1e3,
        "chunks_per_sec": n_chunks / dt,
        "ms_per_chunk": dt / n_chunks * 1e3,
        "total_spikes": total_spikes,
    }

# ─── Main ───

def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="QC Benchmark Suite")
    parser.add_argument("--quick", action="store_true", help="Reduced iterations")
    parser.add_argument("--json", action="store_true", help="JSON output")
    args = parser.parse_args()

    n_calls = 1_000 if args.quick else 10_000
    sizes = [32, 128, 256]
    results: dict[str, Any] = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}

    if not args.json:
        print(f"\n\033[1;36m{'='*60}")
        print("  Quantum Cognition Benchmark Suite")
        print(f"{'='*60}\033[0m\n")
        print(f"  Iterations: {n_calls}  |  Sizes: {sizes}")
        print(f"  System: {os.uname().sysname} {os.uname().machine}\n")

    # 1. apply_measurement
    if not args.json:
        print("\033[1m1. SpinPoolMPS.apply_measurement\033[0m")
        print(f"  {'Sites':>6}  {'Calls':>8}  {'Total (ms)':>12}  {'Per-call (µs)':>14}")
    am_results = {}
    for s in sizes:
        us = bench_py_apply_measurement(s, n_calls)
        total_ms = us * n_calls / 1e3
        am_results[s] = us
        if not args.json:
            print(f"  {s:>6}  {n_calls:>8}  {total_ms:>12.1f}  {us:>14.2f}")
    results["apply_measurement_us"] = am_results

    # 2. neuron_step
    if not args.json:
        print("\n\033[1m2. Neuron Step (batched)\033[0m")
        print(f"  {'Neurons':>8}  {'Steps':>8}  {'Total (ms)':>12}  {'Per-step (µs)':>14}")
    ns_results = {}
    for s in sizes:
        steps = n_calls // 10
        us = bench_py_neuron_step(s, steps)
        total_ms = us * steps * s / 1e3
        ns_results[s] = us
        if not args.json:
            print(f"  {s:>8}  {steps:>8}  {total_ms:>12.1f}  {us:>14.2f}")
    results["neuron_step_us"] = ns_results

    # 3. singlet_yield
    if not args.json:
        print("\n\033[1m3. Radical Pair — singlet_yield\033[0m")
    us = bench_py_singlet_yield(n_calls)
    results["singlet_yield_us"] = us
    if not args.json:
        print(f"  {n_calls} calls: {us:.2f} µs/call ({us*n_calls/1e3:.1f} ms total)")

    # 4. coupling_matrix
    if not args.json:
        print("\n\033[1m4. Kane — coupling_matrix\033[0m")
        print(f"  {'Sites':>6}  {'Time (ms)':>12}")
    cm_results = {}
    for s in sizes:
        ms = bench_py_coupling_matrix(s)
        cm_results[s] = ms
        if not args.json:
            print(f"  {s:>6}  {ms:>12.2f}")
    results["coupling_matrix_ms"] = cm_results

    # 5. E2E pipeline
    if not args.json:
        print("\n\033[1m5. E2E Pipeline Throughput\033[0m")
    e2e = bench_e2e_pipeline(32, 200 if not args.quick else 50)
    results["e2e_pipeline"] = e2e
    if not args.json:
        print(f"  {e2e['chunks_per_sec']:.0f} chunks/sec  "
              f"({e2e['ms_per_chunk']:.1f} ms/chunk, "
              f"{e2e['total_spikes']} spikes)")

    # 6. Rust benchmarks
    if not args.json:
        print("\n\033[1m6. Rust Benchmarks\033[0m")
    for rs_file in ["spin_pool.rs", "radical_pair.rs", "kane_mapper.rs"]:
        rust_out = _run_rust_benchmark(rs_file)
        if rust_out:
            results[f"rust_{rs_file}"] = "PASS"
            if not args.json:
                # Extract benchmark lines from output
                for line in rust_out["output"].split("\n"):
                    if "benchmark" in line.lower() or "call" in line.lower():
                        print(f"  [{rs_file}] {line.strip()}")
        else:
            results[f"rust_{rs_file}"] = "SKIP"
            if not args.json:
                print(f"  [{rs_file}] \033[33mSKIPPED\033[0m")

    if not args.json:
        print(f"\n\033[1;32m{'='*60}")
        print("  Benchmark Complete")
        print(f"{'='*60}\033[0m\n")

    if args.json:
        print(json.dumps(results, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
