"""
SC-NeuroCore v3 - Formal Benchmark Report Generator
====================================================

Runs head-to-head benchmarks between v2 (Python/NumPy) and v3 (Rust)
for all operations specified in the V3 Migration Blueprint section 8.

Includes both list-based (legacy) and numpy zero-copy variants to
show true kernel performance without FFI marshalling overhead.

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


def bench_pack(n_bits: int = 1_000_000) -> list[dict]:
    """Benchmark pack_bitstream: list vs numpy zero-copy."""
    rng = np.random.RandomState(42)
    bits = rng.randint(0, 2, n_bits).astype(np.uint8)

    v2_time = benchmark(lambda: v2_pack(bits), n_iters=10)
    v3_list_time = benchmark(lambda: v3.pack_bitstream(bits.tolist()), n_iters=10)
    v3_np_time = benchmark(lambda: v3.pack_bitstream_numpy(bits), n_iters=10)

    return [
        {
            "operation": f"pack (list, {n_bits // 1000}K)",
            "v2_ms": v2_time / 10 * 1000,
            "v3_ms": v3_list_time / 10 * 1000,
            "speedup": fmt_speedup(v2_time, v3_list_time),
            "target": "6x",
        },
        {
            "operation": f"pack (numpy, {n_bits // 1000}K)",
            "v2_ms": v2_time / 10 * 1000,
            "v3_ms": v3_np_time / 10 * 1000,
            "speedup": fmt_speedup(v2_time, v3_np_time),
            "target": "6x",
        },
    ]


def bench_popcount(n_words: int = 1_000_000) -> list[dict]:
    """Benchmark popcount: list vs numpy zero-copy."""
    rng = np.random.RandomState(42)
    bits = rng.randint(0, 2, n_words * 64).astype(np.uint8)
    packed_v2 = v2_pack(bits)
    packed_np = np.asarray(v3.pack_bitstream_numpy(bits))

    v2_time = benchmark(lambda: v2_popcount(packed_v2), n_iters=10)
    v3_list_time = benchmark(lambda: v3.popcount(packed_v2.tolist()), n_iters=10)
    v3_np_time = benchmark(lambda: v3.popcount_numpy(packed_np), n_iters=10)

    return [
        {
            "operation": f"popcount (list, {n_words // 1000}K)",
            "v2_ms": v2_time / 10 * 1000,
            "v3_ms": v3_list_time / 10 * 1000,
            "speedup": fmt_speedup(v2_time, v3_list_time),
            "target": "20x",
        },
        {
            "operation": f"popcount (numpy, {n_words // 1000}K)",
            "v2_ms": v2_time / 10 * 1000,
            "v3_ms": v3_np_time / 10 * 1000,
            "speedup": fmt_speedup(v2_time, v3_np_time),
            "target": "20x",
        },
    ]


def bench_dense_forward(n_in: int = 64, n_out: int = 32, length: int = 1024) -> list[dict]:
    """Benchmark dense forward variants."""
    rng = np.random.RandomState(42)
    inputs = rng.uniform(0, 1, n_in)
    inputs_f64 = inputs.astype(np.float64)

    v2_layer = V2Layer(n_inputs=n_in, n_neurons=n_out, length=length)
    v3_layer = V3Layer(n_inputs=n_in, n_neurons=n_out, length=length)
    packed_inputs = v3.batch_encode_numpy(inputs_f64, length=length, seed=42)

    # Warm-up once to stabilize caches and rayon thread-pool initialization.
    v2_layer.forward(inputs)
    v3_layer.forward(inputs)
    v3_layer.forward_fast(inputs)
    v3_layer.forward_prepacked(packed_inputs)
    v3_layer.forward_prepacked_numpy(packed_inputs)
    v3_layer.forward_numpy(inputs_f64)

    v2_time = benchmark(lambda: v2_layer.forward(inputs), n_iters=10)
    v3_time = benchmark(lambda: v3_layer.forward(inputs), n_iters=10)
    v3_fast_time = benchmark(lambda: v3_layer.forward_fast(inputs), n_iters=10)
    v3_prepacked_time = benchmark(lambda: v3_layer.forward_prepacked(packed_inputs), n_iters=10)
    v3_prepacked_numpy_time = benchmark(
        lambda: v3_layer.forward_prepacked_numpy(packed_inputs), n_iters=10
    )
    v3_numpy_time = benchmark(lambda: v3_layer.forward_numpy(inputs_f64), n_iters=10)

    return [
        {
            "operation": f"dense forward ({n_in}->{n_out}, L={length})",
            "v2_ms": v2_time / 10 * 1000,
            "v3_ms": v3_time / 10 * 1000,
            "speedup": fmt_speedup(v2_time, v3_time),
            "target": "70x",
        },
        {
            "operation": f"dense fast ({n_in}->{n_out}, L={length})",
            "v2_ms": v2_time / 10 * 1000,
            "v3_ms": v3_fast_time / 10 * 1000,
            "speedup": fmt_speedup(v2_time, v3_fast_time),
            "target": "70x",
        },
        {
            "operation": f"dense prepacked ({n_in}->{n_out}, L={length})",
            "v2_ms": v2_time / 10 * 1000,
            "v3_ms": v3_prepacked_time / 10 * 1000,
            "speedup": fmt_speedup(v2_time, v3_prepacked_time),
            "target": "70x",
        },
        {
            "operation": f"dense prepacked numpy ({n_in}->{n_out}, L={length})",
            "v2_ms": v2_time / 10 * 1000,
            "v3_ms": v3_prepacked_numpy_time / 10 * 1000,
            "speedup": fmt_speedup(v2_time, v3_prepacked_numpy_time),
            "target": "70x",
        },
        {
            "operation": f"dense numpy ({n_in}->{n_out}, L={length})",
            "v2_ms": v2_time / 10 * 1000,
            "v3_ms": v3_numpy_time / 10 * 1000,
            "speedup": fmt_speedup(v2_time, v3_numpy_time),
            "target": "70x",
        },
    ]


def bench_dense_fused(n_in: int = 64, n_out: int = 32, length: int = 1024) -> list[dict]:
    """Benchmark dense fused forward path explicitly."""
    rng = np.random.RandomState(42)
    inputs = rng.uniform(0, 1, n_in)

    v2_layer = V2Layer(n_inputs=n_in, n_neurons=n_out, length=length)
    v3_layer = v3.DenseLayer(n_in, n_out, length)

    # Warm-up
    v2_layer.forward(inputs)
    v3_layer.forward_fast(inputs.tolist(), 42)

    v2_time = benchmark(lambda: v2_layer.forward(inputs), n_iters=10)
    v3_fused_time = benchmark(lambda: v3_layer.forward_fast(inputs.tolist(), 42), n_iters=10)

    return [
        {
            "operation": f"dense fused ({n_in}->{n_out}, L={length})",
            "v2_ms": v2_time / 10 * 1000,
            "v3_ms": v3_fused_time / 10 * 1000,
            "speedup": fmt_speedup(v2_time, v3_fused_time),
            "target": "70x",
        }
    ]


def bench_dense_batch(
    n_samples: int = 100, n_in: int = 64, n_out: int = 32, length: int = 1024
) -> list[dict]:
    """Benchmark batched dense forward (single FFI call for N samples)."""
    rng = np.random.RandomState(42)
    inputs_batch = rng.uniform(0, 1, (n_samples, n_in)).astype(np.float64)

    v2_layer = V2Layer(n_inputs=n_in, n_neurons=n_out, length=length)
    v3_layer = V3Layer(n_inputs=n_in, n_neurons=n_out, length=length)

    # Warm-up
    _ = v2_layer.forward(inputs_batch[0])
    _ = v3_layer.forward_batch_numpy(inputs_batch)

    def run_v2_batch():
        for row in inputs_batch:
            v2_layer.forward(row)

    def run_v3_batch():
        return v3_layer.forward_batch_numpy(inputs_batch)

    v2_time = benchmark(run_v2_batch)
    v3_time = benchmark(run_v3_batch)

    return [
        {
            "operation": f"dense batch ({n_samples}x{n_in}->{n_out}, L={length})",
            "v2_ms": v2_time * 1000,
            "v3_ms": v3_time * 1000,
            "speedup": fmt_speedup(v2_time, v3_time),
            "target": "70x",
        }
    ]


def bench_lif_step(n_steps: int = 100_000) -> list[dict]:
    """Benchmark LIF neuron step: per-call vs batch."""

    def run_v2():
        lif = V2Lif()
        for _ in range(n_steps):
            lif.step(20, 256, 128, 0)

    def run_v3_percall():
        lif = V3Lif()
        for _ in range(n_steps):
            lif.step(20, 256, 128, 0)

    def run_v3_batch():
        return v3.batch_lif_run(n_steps, leak_k=20, gain_k=256, i_t=128)

    v2_time = benchmark(run_v2)
    v3_percall_time = benchmark(run_v3_percall)
    v3_batch_time = benchmark(run_v3_batch)

    return [
        {
            "operation": f"LIF (per-call, {n_steps // 1000}K)",
            "v2_ms": v2_time * 1000,
            "v3_ms": v3_percall_time * 1000,
            "speedup": fmt_speedup(v2_time, v3_percall_time),
            "target": "400x",
        },
        {
            "operation": f"LIF (batch, {n_steps // 1000}K)",
            "v2_ms": v2_time * 1000,
            "v3_ms": v3_batch_time * 1000,
            "speedup": fmt_speedup(v2_time, v3_batch_time),
            "target": "400x",
        },
    ]


def bench_lif_multi(n_neurons: int = 100, n_steps: int = 100_000) -> list[dict]:
    """Benchmark multi-neuron parallel LIF batch."""
    currents = np.full(n_neurons, 128, dtype=np.int16)

    def run_v2():
        for _ in range(n_neurons):
            lif = V2Lif()
            for _ in range(n_steps):
                lif.step(20, 256, 128, 0)

    def run_v3():
        return v3.batch_lif_run_multi(
            n_neurons,
            n_steps,
            leak_k=20,
            gain_k=256,
            currents=currents,
        )

    v2_time = benchmark(run_v2)
    v3_time = benchmark(run_v3)

    return [
        {
            "operation": f"LIF multi ({n_neurons}x{n_steps // 1000}K)",
            "v2_ms": v2_time * 1000,
            "v3_ms": v3_time * 1000,
            "speedup": fmt_speedup(v2_time, v3_time),
            "target": "400x",
        }
    ]


def main():
    print("SC-NeuroCore v3 - Benchmark Report")
    print("=" * 90)
    print(f"Platform: {sys.platform}")
    print(f"SIMD tier: {v3.simd_tier()}")
    print(f"v3 version: {v3.__version__}")
    print()

    results = []
    results.extend(bench_pack())
    results.extend(bench_popcount())
    results.extend(bench_dense_forward())
    results.extend(bench_dense_fused())
    results.extend(bench_dense_batch())
    results.extend(bench_lif_step())
    results.extend(bench_lif_multi())

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
    print("'list' variants cross Python/Rust FFI via list->Vec conversion (2 copies).")
    print("'numpy' variants use PyReadonlyArray for zero-copy buffer access.")
    print("'batch' variants process entire arrays in a single FFI call.")
    print("Dense forward uses rayon parallelism across neurons.")
    print("'fast' variants use per-input parallel encoding with rayon.")
    print("'prepacked' variants skip encoding entirely (pre-encoded inputs).")
    print("'dense numpy' variant runs in one FFI call with numpy input/output.")

    return results


if __name__ == "__main__":
    main()
