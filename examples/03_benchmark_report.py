"""
SC-NeuroCore v3 - Formal Benchmark Report Generator
====================================================

Runs head-to-head benchmarks between v2 (Python/NumPy) and v3 (Rust)
for all operations specified in the V3 Migration Blueprint section 8.

Usage:
    cd 03_CODE/sc-neurocore
    $env:PYTHONPATH='src'
    .\\.venv\\Scripts\\python examples/03_benchmark_report.py
"""

from __future__ import annotations

import sys
import time

import numpy as np

# -- v2 imports --
from sc_neurocore.accel.vector_ops import (
    pack_bitstream as v2_pack,
    vec_popcount as v2_popcount,
)
from sc_neurocore.neurons import FixedPointLIFNeuron as V2Lif
from sc_neurocore.layers import VectorizedSCLayer as V2Layer

# -- v3 imports --
import sc_neurocore_engine as v3
from sc_neurocore_engine import FixedPointLIFNeuron as V3Lif
from sc_neurocore_engine.layers import VectorizedSCLayer as V3Layer


def benchmark(fn, n_iters: int = 1) -> float:
    """Time a function call, return seconds."""
    start = time.perf_counter()
    for _ in range(n_iters):
        fn()
    elapsed = time.perf_counter() - start
    return elapsed


def fmt_speedup(v2_time: float, v3_time: float) -> str:
    if v3_time == 0:
        return "inf"
    ratio = v2_time / v3_time
    return f"{ratio:.1f}x"


def bench_pack(n_bits: int = 1_000_000) -> dict:
    """Benchmark pack_bitstream for 1M bits."""
    rng = np.random.RandomState(42)
    bits = rng.randint(0, 2, n_bits).astype(np.uint8)

    v2_time = benchmark(lambda: v2_pack(bits), n_iters=10)
    v3_time = benchmark(lambda: v3.pack_bitstream(bits.tolist()), n_iters=10)

    return {
        "operation": f"pack_bitstream ({n_bits // 1000}K bits)",
        "v2_ms": v2_time / 10 * 1000,
        "v3_ms": v3_time / 10 * 1000,
        "speedup": fmt_speedup(v2_time, v3_time),
        "target": "6x",
    }


def bench_popcount(n_words: int = 1_000_000) -> dict:
    """Benchmark popcount for 1M u64 words."""
    rng = np.random.RandomState(42)
    bits = rng.randint(0, 2, n_words * 64).astype(np.uint8)
    packed = v2_pack(bits)

    v2_time = benchmark(lambda: v2_popcount(packed), n_iters=10)
    v3_time = benchmark(lambda: v3.popcount(packed.tolist()), n_iters=10)

    return {
        "operation": f"popcount ({n_words // 1000}K words)",
        "v2_ms": v2_time / 10 * 1000,
        "v3_ms": v3_time / 10 * 1000,
        "speedup": fmt_speedup(v2_time, v3_time),
        "target": "20x",
    }


def bench_dense_forward(n_in: int = 64, n_out: int = 32, length: int = 1024) -> dict:
    """Benchmark dense forward pass."""
    rng = np.random.RandomState(42)
    inputs = rng.uniform(0, 1, n_in)

    v2_layer = V2Layer(n_inputs=n_in, n_neurons=n_out, length=length)
    v3_layer = V3Layer(n_inputs=n_in, n_neurons=n_out, length=length)

    v2_time = benchmark(lambda: v2_layer.forward(inputs), n_iters=10)
    v3_time = benchmark(lambda: v3_layer.forward(inputs), n_iters=10)

    return {
        "operation": f"dense forward ({n_in}->{n_out}, L={length})",
        "v2_ms": v2_time / 10 * 1000,
        "v3_ms": v3_time / 10 * 1000,
        "speedup": fmt_speedup(v2_time, v3_time),
        "target": "70x",
    }


def bench_lif_step(n_steps: int = 100_000) -> dict:
    """Benchmark LIF neuron step execution."""
    v2_lif = V2Lif()
    v3_lif = V3Lif()

    def run_v2():
        lif = V2Lif()
        for _ in range(n_steps):
            lif.step(20, 256, 128, 0)

    def run_v3():
        lif = V3Lif()
        for _ in range(n_steps):
            lif.step(20, 256, 128, 0)

    v2_time = benchmark(run_v2)
    v3_time = benchmark(run_v3)

    return {
        "operation": f"LIF step ({n_steps // 1000}K steps)",
        "v2_ms": v2_time * 1000,
        "v3_ms": v3_time * 1000,
        "speedup": fmt_speedup(v2_time, v3_time),
        "target": "400x",
    }


def main():
    print("SC-NeuroCore v3 - Benchmark Report")
    print("=" * 70)
    print(f"Platform: {sys.platform}")
    print(f"SIMD tier: {v3.simd_tier()}")
    print(f"v3 version: {v3.__version__}")
    print()

    results = [
        bench_pack(),
        bench_popcount(),
        bench_dense_forward(),
        bench_lif_step(),
    ]

    # Print table
    print(f"{'Operation':<40} {'v2 (ms)':<12} {'v3 (ms)':<12} {'Speedup':<10} {'Target':<10}")
    print("-" * 84)
    for r in results:
        print(
            f"{r['operation']:<40} "
            f"{r['v2_ms']:<12.3f} "
            f"{r['v3_ms']:<12.3f} "
            f"{r['speedup']:<10} "
            f"{r['target']:<10}"
        )

    print()
    print("Note: Targets from V3_MIGRATION_BLUEPRINT.md section 8.")
    print("SIMD tier affects popcount and pack performance significantly.")
    print("Benchmarks run single-threaded; rayon parallelism adds 4-16x on multi-core.")

    return results


if __name__ == "__main__":
    main()
