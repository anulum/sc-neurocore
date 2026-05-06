# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Quantum Cognition Performance Benchmark Suite
"""
Comprehensive benchmarks for the quantum cognition module.

Profiles all hot paths across Python, measures per-call latencies,
and runs end-to-end population simulations at multiple scales.

Usage:
    PYTHONPATH=src python benchmarks/bench_quantum_cognition.py
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Ensure PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sc_neurocore.quantum_cognition import (
    SpinPoolMPS,
    HybridFisherPosnerLIF,
    FisherPosnerQuantumBridge,
    GOTMBrain,
    embed_chunks,
    ContentChunk,
    index_gotm_repo,
)


@dataclass
class BenchmarkResult:
    """Single benchmark measurement."""

    name: str
    scale: str
    total_time_ms: float
    iterations: int
    per_call_us: float
    throughput: str = ""
    notes: str = ""


def _fmt(val: float) -> str:
    if val < 1.0:
        return f"{val * 1000:.1f} ns"
    elif val < 1000.0:
        return f"{val:.2f} µs"
    else:
        return f"{val / 1000:.2f} ms"


# ─── Individual kernel benchmarks ───


def bench_spin_pool_measurement(n_sites: int, n_steps: int) -> BenchmarkResult:
    pool = SpinPoolMPS(n_sites=n_sites)
    t0 = time.perf_counter()
    for i in range(n_steps):
        pool.apply_measurement(i % n_sites, 1.0)
    elapsed = time.perf_counter() - t0
    return BenchmarkResult(
        name="SpinPoolMPS.apply_measurement",
        scale=f"{n_sites} sites × {n_steps} calls",
        total_time_ms=elapsed * 1000,
        iterations=n_steps,
        per_call_us=elapsed / n_steps * 1e6,
    )


def bench_atp_efficiency(n_sites: int, n_calls: int) -> BenchmarkResult:
    pool = SpinPoolMPS(n_sites=n_sites)
    # Warm up entanglement map
    for i in range(100):
        pool.apply_measurement(i % n_sites, 1.0)
    t0 = time.perf_counter()
    for i in range(n_calls):
        pool.get_local_atp_efficiency(i % n_sites)
    elapsed = time.perf_counter() - t0
    return BenchmarkResult(
        name="SpinPoolMPS.get_local_atp_efficiency",
        scale=f"{n_sites} sites × {n_calls} calls",
        total_time_ms=elapsed * 1000,
        iterations=n_calls,
        per_call_us=elapsed / n_calls * 1e6,
    )


def bench_neuron_step(n_neurons: int, n_steps: int) -> BenchmarkResult:
    pool = SpinPoolMPS(n_sites=n_neurons)
    neurons = [HybridFisherPosnerLIF(i, pool) for i in range(n_neurons)]
    rng = np.random.default_rng(42)
    total_spikes = 0

    t0 = time.perf_counter()
    for step in range(n_steps):
        currents = rng.uniform(15, 35, n_neurons)
        for i, neuron in enumerate(neurons):
            _, spiked = neuron.step(float(currents[i]))
            if spiked:
                total_spikes += 1
    elapsed = time.perf_counter() - t0
    total_calls = n_neurons * n_steps
    return BenchmarkResult(
        name="HybridFisherPosnerLIF.step",
        scale=f"{n_neurons} neurons × {n_steps} steps",
        total_time_ms=elapsed * 1000,
        iterations=total_calls,
        per_call_us=elapsed / total_calls * 1e6,
        notes=f"{total_spikes} spikes",
    )


def bench_embed_chunks(n_chunks: int, n_dims: int) -> BenchmarkResult:
    chunks = [
        ContentChunk(
            "BENCH",
            f"file_{i}.py",
            0,
            f"benchmark content {i} with words " * 30,
            "code",
            1.0,
        )
        for i in range(n_chunks)
    ]
    t0 = time.perf_counter()
    vecs = embed_chunks(chunks, n_dims=n_dims)
    elapsed = time.perf_counter() - t0
    return BenchmarkResult(
        name="embed_chunks",
        scale=f"{n_chunks} chunks × {n_dims} dims",
        total_time_ms=elapsed * 1000,
        iterations=n_chunks,
        per_call_us=elapsed / n_chunks * 1e6,
    )


def bench_bridge_sync(n_qubits: int, n_calls: int) -> BenchmarkResult:
    bridge = FisherPosnerQuantumBridge(n_qubits, backend="emulated")
    pairs = [(i, (i + 1) % n_qubits) for i in range(0, n_qubits - 1, 2)]
    t0 = time.perf_counter()
    for _ in range(n_calls):
        bridge.execute_non_local_sync(pairs)
    elapsed = time.perf_counter() - t0
    return BenchmarkResult(
        name="FisherPosnerQuantumBridge.execute_non_local_sync",
        scale=f"{n_qubits} qubits, {len(pairs)} pairs × {n_calls} calls",
        total_time_ms=elapsed * 1000,
        iterations=n_calls,
        per_call_us=elapsed / n_calls * 1e6,
    )


# ─── End-to-end benchmarks ───


def bench_e2e_population(n_neurons: int, n_steps: int) -> BenchmarkResult:
    """Full population simulation: spin pool + all neurons + quantum feedback."""
    pool = SpinPoolMPS(n_sites=n_neurons)
    bridge = FisherPosnerQuantumBridge(min(n_neurons, 20), backend="emulated")
    neurons = [HybridFisherPosnerLIF(i, pool) for i in range(n_neurons)]
    rng = np.random.default_rng(42)
    total_spikes = 0

    t0 = time.perf_counter()
    for step in range(n_steps):
        # Phase optimisation every 100 steps
        if step % 100 == 0:
            bridge.optimize_phases(target_coherence=0.7, learning_rate=0.1, n_steps=1)

        currents = rng.uniform(15, 35, n_neurons)
        for i, neuron in enumerate(neurons):
            _, spiked = neuron.step(float(currents[i]))
            if spiked:
                total_spikes += 1
    elapsed = time.perf_counter() - t0

    throughput = f"{n_neurons * n_steps / elapsed:.0f} neuron-steps/s"
    return BenchmarkResult(
        name="E2E Population Simulation",
        scale=f"{n_neurons} neurons × {n_steps} steps",
        total_time_ms=elapsed * 1000,
        iterations=n_neurons * n_steps,
        per_call_us=elapsed / (n_neurons * n_steps) * 1e6,
        throughput=throughput,
        notes=f"{total_spikes} spikes, bridge every 100 steps",
    )


def bench_e2e_gotm_learning(max_chunks: int) -> BenchmarkResult:
    """Full GOTM Brain learning loop on SC-NEUROCORE docs."""
    repo_path = str(Path(__file__).resolve().parent.parent)
    brain = GOTMBrain(n_neurons=32, bridge_backend="emulated", seed=42)

    t0 = time.perf_counter()
    steps = brain.learn_from_repo(
        repo_path + "/docs",
        repo_name="SC-NEUROCORE/docs",
        max_chunks=max_chunks,
    )
    elapsed = time.perf_counter() - t0

    total_spikes = sum(s.n_spikes for s in steps)
    return BenchmarkResult(
        name="E2E GOTMBrain.learn_from_repo",
        scale=f"{len(steps)} chunks, 32 neurons",
        total_time_ms=elapsed * 1000,
        iterations=len(steps),
        per_call_us=elapsed / max(len(steps), 1) * 1e6,
        throughput=f"{len(steps) / max(elapsed, 0.001):.0f} chunks/s",
        notes=f"{total_spikes} spikes total",
    )


def bench_e2e_index_repo() -> BenchmarkResult:
    """Index SC-NEUROCORE source tree."""
    repo_path = str(Path(__file__).resolve().parent.parent / "src")
    t0 = time.perf_counter()
    chunks = index_gotm_repo(repo_path, "SC-NEUROCORE")
    elapsed = time.perf_counter() - t0
    return BenchmarkResult(
        name="E2E index_gotm_repo (SC-NEUROCORE/src)",
        scale=f"{len(chunks)} chunks indexed",
        total_time_ms=elapsed * 1000,
        iterations=len(chunks),
        per_call_us=elapsed / max(len(chunks), 1) * 1e6,
        throughput=f"{len(chunks) / max(elapsed, 0.001):.0f} chunks/s",
    )


# ─── Main runner ───


def main() -> None:
    print("=" * 72)
    print("  SC-NeuroCore Quantum Cognition — Performance Benchmark Suite")
    print("=" * 72)
    print()

    results: list[BenchmarkResult] = []

    # ─── Kernel benchmarks ───
    print(">>> Kernel Benchmarks")
    print("-" * 72)

    for n_sites in (32, 128, 256):
        r = bench_spin_pool_measurement(n_sites, 10000)
        results.append(r)
        print(f"  {r.name} [{r.scale}]: {_fmt(r.per_call_us)}/call ({r.total_time_ms:.1f} ms)")

    for n_sites in (128, 256):
        r = bench_atp_efficiency(n_sites, 100000)
        results.append(r)
        print(f"  {r.name} [{r.scale}]: {_fmt(r.per_call_us)}/call ({r.total_time_ms:.1f} ms)")

    for n_neurons, n_steps in [(32, 1000), (128, 1000), (256, 500)]:
        r = bench_neuron_step(n_neurons, n_steps)
        results.append(r)
        print(
            f"  {r.name} [{r.scale}]: {_fmt(r.per_call_us)}/call ({r.total_time_ms:.1f} ms) [{r.notes}]"
        )

    for n_chunks in (100, 1000, 5000):
        r = bench_embed_chunks(n_chunks, 32)
        results.append(r)
        print(f"  {r.name} [{r.scale}]: {_fmt(r.per_call_us)}/call ({r.total_time_ms:.1f} ms)")

    for n_qubits in (4, 8, 16):
        r = bench_bridge_sync(n_qubits, 1000)
        results.append(r)
        print(f"  {r.name} [{r.scale}]: {_fmt(r.per_call_us)}/call ({r.total_time_ms:.1f} ms)")

    # ─── E2E benchmarks ───
    print()
    print(">>> End-to-End Benchmarks")
    print("-" * 72)

    for n_neurons, n_steps in [(32, 1000), (128, 1000), (256, 500), (512, 200)]:
        r = bench_e2e_population(n_neurons, n_steps)
        results.append(r)
        print(f"  {r.name} [{r.scale}]: {r.total_time_ms:.0f} ms | {r.throughput} [{r.notes}]")

    r = bench_e2e_index_repo()
    results.append(r)
    print(f"  {r.name}: {r.total_time_ms:.0f} ms | {r.throughput}")

    for max_chunks in (20, 50):
        r = bench_e2e_gotm_learning(max_chunks)
        results.append(r)
        print(f"  {r.name} [{r.scale}]: {r.total_time_ms:.0f} ms | {r.throughput} [{r.notes}]")

    # ─── Summary table ───
    print()
    print("=" * 72)
    print("  SUMMARY")
    print("=" * 72)
    print(f"{'Benchmark':<45} {'Per-call':>12} {'Total':>10}")
    print("-" * 72)
    for r in results:
        print(f"  {r.name[:42]:<43} {_fmt(r.per_call_us):>12} {r.total_time_ms:>8.1f} ms")

    # ─── Save JSON ───
    out_path = (
        Path(__file__).resolve().parent.parent
        / "docs"
        / "internal"
        / "BENCHMARK_QUANTUM_COGNITION.json"
    )
    with open(out_path, "w") as f:
        json.dump(
            {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "python_version": sys.version.split()[0],
                "results": [
                    {
                        "name": r.name,
                        "scale": r.scale,
                        "total_ms": round(r.total_time_ms, 2),
                        "per_call_us": round(r.per_call_us, 3),
                        "throughput": r.throughput,
                        "notes": r.notes,
                    }
                    for r in results
                ],
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
