# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Performance Benchmark Suite

#!/usr/bin/env python3
"""
SC-NeuroCore Performance Benchmark Suite
=========================================

Measures throughput and accuracy across core stochastic computing operations:
  1. Bitstream encoding (LFSR + comparator)
  2. Pack / unpack uint64 operations
  3. Dense layer forward pass (packed bitwise MAC)
  4. GPU vs CPU comparison (when CuPy is available)
  5. Fixed-point LIF neuron stepping
  6. Full pipeline (encode -> synapse -> dot-product -> neuron)
  7. Bitstream length scaling (wall time vs L for 32x16 dense)
  8. Memory footprint tracking (tracemalloc peak allocation)

Usage::

    python scripts/benchmark_suite.py            # quick mode (default)
    python scripts/benchmark_suite.py --full      # thorough (longer)
    python scripts/benchmark_suite.py --markdown   # output BENCHMARKS.md table
"""

from __future__ import annotations

import argparse
import sys
import os
import time
import tracemalloc
from dataclasses import dataclass
from typing import Callable, List

import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import sc_neurocore.accel.gpu_backend as gpu_backend
from sc_neurocore.accel.vector_ops import pack_bitstream, vec_and, vec_popcount
from sc_neurocore.accel.gpu_backend import (
    xp,
    HAS_CUPY,
    gpu_pack_bitstream,
    gpu_vec_mac,
)
from sc_neurocore.neurons.fixed_point_lif import (
    FixedPointLIFNeuron,
    FixedPointLFSR,
    FixedPointBitstreamEncoder,
)

# ---------------------------------------------------------------------------
# Benchmark harness
# ---------------------------------------------------------------------------


@dataclass
class BenchResult:
    name: str
    iterations: int
    total_sec: float
    throughput: str
    backend: str = "cpu"

    @property
    def avg_us(self) -> float:
        return (self.total_sec / self.iterations) * 1e6

    def __str__(self) -> str:
        return (
            f"{self.name:<45} | {self.backend:<5} | "
            f"{self.iterations:>8} iters | "
            f"{self.avg_us:>10.1f} us/iter | "
            f"{self.throughput}"
        )

    def md_row(self) -> str:
        return (
            f"| {self.name} | {self.backend} | {self.iterations} | "
            f"{self.avg_us:.1f} us | {self.throughput} |"
        )


def _timer(func: Callable[[], object], n_iters: int) -> float:
    """Time *func()* over *n_iters* calls, return total seconds."""
    start = time.perf_counter()
    for _ in range(n_iters):
        func()
    return time.perf_counter() - start


def _gpu_runtime_available() -> bool:
    """Return whether CuPy is installed and its runtime has not failed closed."""
    return HAS_CUPY and not gpu_backend._GPU_RUNTIME_BROKEN


def _gpu_backend_label() -> str:
    """Return a benchmark label for the currently usable GPU backend."""
    return "CuPy" if _gpu_runtime_available() else "NumPy fallback"


def _gpu_banner_label() -> str:
    """Return a top-level backend label without claiming CUDA before first use."""
    if HAS_CUPY and not gpu_backend._GPU_RUNTIME_BROKEN:
        return "CuPy import detected; runtime checked lazily"
    return "NumPy (CPU only)"


def _synchronize_gpu_if_available() -> None:
    """Synchronize CUDA streams only while the backend is still genuinely live."""
    if _gpu_runtime_available():
        xp.cuda.Stream.null.synchronize()


# ---------------------------------------------------------------------------
# Individual benchmarks
# ---------------------------------------------------------------------------


def bench_lfsr(n_iters: int) -> BenchResult:
    lfsr = FixedPointLFSR(seed=0xACE1)
    t = _timer(lfsr.step, n_iters)
    msteps = n_iters / t / 1e6
    return BenchResult("LFSR step (16-bit)", n_iters, t, f"{msteps:.2f} Mstep/s")


def bench_encoder(n_iters: int) -> BenchResult:
    enc = FixedPointBitstreamEncoder(seed_init=0xACE1)
    x = 32768

    def step() -> None:
        enc.step(x)

    t = _timer(step, n_iters)
    msteps = n_iters / t / 1e6
    return BenchResult("Bitstream encoder step", n_iters, t, f"{msteps:.2f} Mstep/s")


def bench_pack_1d(length: int, n_iters: int) -> BenchResult:
    bits = np.random.randint(0, 2, size=length, dtype=np.uint8)

    def run() -> None:
        pack_bitstream(bits)

    t = _timer(run, n_iters)
    gbps = (length * n_iters) / t / 1e9
    return BenchResult(f"pack_bitstream 1-D ({length})", n_iters, t, f"{gbps:.2f} Gbit/s")


def bench_pack_2d(batch: int, length: int, n_iters: int) -> BenchResult:
    bits = np.random.randint(0, 2, size=(batch, length), dtype=np.uint8)

    def run() -> None:
        pack_bitstream(bits)

    t = _timer(run, n_iters)
    gbps = (batch * length * n_iters) / t / 1e9
    return BenchResult(f"pack_bitstream 2-D ({batch}x{length})", n_iters, t, f"{gbps:.2f} Gbit/s")


def bench_vec_and(n_words: int, n_iters: int) -> BenchResult:
    a = np.random.randint(0, 2**63, size=n_words, dtype=np.uint64)
    b = np.random.randint(0, 2**63, size=n_words, dtype=np.uint64)

    def run() -> None:
        vec_and(a, b)

    t = _timer(run, n_iters)
    gbps = (n_words * 64 * n_iters) / t / 1e9
    return BenchResult(f"vec_and ({n_words} words)", n_iters, t, f"{gbps:.2f} Gbit/s")


def bench_popcount(n_words: int, n_iters: int) -> BenchResult:
    a = np.random.randint(0, 2**63, size=n_words, dtype=np.uint64)

    def run() -> None:
        vec_popcount(a)

    t = _timer(run, n_iters)
    gbps = (n_words * 64 * n_iters) / t / 1e9
    return BenchResult(f"vec_popcount SWAR ({n_words} words)", n_iters, t, f"{gbps:.2f} Gbit/s")


def bench_lif(n_iters: int) -> BenchResult:
    neuron = FixedPointLIFNeuron()

    def run() -> None:
        neuron.step(20, 256, 128, 0)

    t = _timer(run, n_iters)
    msteps = n_iters / t / 1e6
    return BenchResult("LIF neuron step (Q8.8)", n_iters, t, f"{msteps:.2f} Mstep/s")


def bench_dense_forward(n_neurons: int, n_inputs: int, length: int, n_iters: int) -> BenchResult:
    from sc_neurocore.layers.vectorized_layer import VectorizedSCLayer

    layer = VectorizedSCLayer(n_inputs=n_inputs, n_neurons=n_neurons, length=length)
    x = np.random.uniform(0, 1, n_inputs).tolist()

    def run() -> None:
        layer.forward(x)

    t = _timer(run, n_iters)
    ops_per_iter = n_neurons * n_inputs * length
    gops = (ops_per_iter * n_iters) / t / 1e9
    return BenchResult(
        f"Dense forward ({n_neurons}x{n_inputs}, L={length})",
        n_iters,
        t,
        f"{gops:.2f} GOP/s (SC)",
    )


def bench_pipeline(n_inputs: int, n_steps: int, n_iters: int) -> BenchResult:
    """Full pipeline: LFSR encode -> AND synapse -> popcount -> LIF."""

    def run() -> None:
        input_encs = [FixedPointBitstreamEncoder(seed_init=0xACE1 + i * 7) for i in range(n_inputs)]
        weight_encs = [
            FixedPointBitstreamEncoder(seed_init=0xBEEF + i * 13) for i in range(n_inputs)
        ]
        neuron = FixedPointLIFNeuron(refractory_period=0)
        x_vals = [52428] * n_inputs  # ~0.8
        w_vals = [52428] * n_inputs

        for _ in range(n_steps):
            pre = [e.step(x) for e, x in zip(input_encs, x_vals)]
            wbits = [e.step(w) for e, w in zip(weight_encs, w_vals)]
            post = [p & w for p, w in zip(pre, wbits)]
            count = sum(post)
            I_t = (256 * count) // n_inputs if n_inputs else 0
            neuron.step(10, 256, I_t, 0)

    t = _timer(run, n_iters)
    total_steps = n_steps * n_iters
    ksteps = total_steps / t / 1e3
    return BenchResult(
        f"Full pipeline ({n_inputs} syn, {n_steps} steps)",
        n_iters,
        t,
        f"{ksteps:.1f} Kstep/s",
    )


# ---------------------------------------------------------------------------
# GPU benchmarks (CuPy)
# ---------------------------------------------------------------------------


def bench_gpu_pack(length: int, n_iters: int) -> BenchResult:
    bits = xp.random.randint(0, 2, size=length, dtype=xp.uint8)

    def run() -> None:
        gpu_pack_bitstream(bits)
        _synchronize_gpu_if_available()

    t = _timer(run, n_iters)
    gbps = (length * n_iters) / t / 1e9
    return BenchResult(
        f"gpu_pack_bitstream ({length})",
        n_iters,
        t,
        f"{gbps:.2f} Gbit/s",
        backend="gpu" if _gpu_runtime_available() else "cpu",
    )


def bench_gpu_mac(n_neurons: int, n_inputs: int, n_words: int, n_iters: int) -> BenchResult:
    w = xp.random.randint(0, 2**63, size=(n_neurons, n_inputs, n_words), dtype=xp.uint64)
    inp = xp.random.randint(0, 2**63, size=(n_inputs, n_words), dtype=xp.uint64)

    def run() -> None:
        gpu_vec_mac(w, inp)
        _synchronize_gpu_if_available()

    t = _timer(run, n_iters)
    ops = n_neurons * n_inputs * n_words * 64 * n_iters
    gops = ops / t / 1e9
    return BenchResult(
        f"gpu_vec_mac ({n_neurons}x{n_inputs}x{n_words}w)",
        n_iters,
        t,
        f"{gops:.2f} GOP/s",
        backend="gpu" if _gpu_runtime_available() else "cpu",
    )


# ---------------------------------------------------------------------------
# Bitstream length scaling
# ---------------------------------------------------------------------------


def bench_bitstream_length_scaling(n_runs: int = 5) -> List[BenchResult]:
    """Wall time and throughput for dense forward at varying bitstream lengths.

    Fixed network: 32 inputs, 16 neurons. Measures whether cost scales
    linearly with bitstream length L.
    """
    from sc_neurocore.layers.vectorized_layer import VectorizedSCLayer

    N_INPUTS, N_NEURONS = 32, 16
    lengths = [128, 256, 512, 1024, 2048, 4096]
    results: List[BenchResult] = []

    print(f"\n{'Config':<28} | {'Mean Time (ms)':>14} | {'Throughput (Mbit/s)':>20}")
    print("-" * 70)

    for L in lengths:
        layer = VectorizedSCLayer(n_inputs=N_INPUTS, n_neurons=N_NEURONS, length=L)
        x = np.random.uniform(0, 1, N_INPUTS).tolist()

        # warmup
        layer.forward(x)

        times: list[float] = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            layer.forward(x)
            times.append(time.perf_counter() - t0)

        mean_s = float(np.mean(times))
        total_bits = N_NEURONS * N_INPUTS * L
        throughput_mbps = total_bits / mean_s / 1e6

        print(f"  32x16, L={L:<5d}            | {mean_s * 1e3:>12.2f}ms | {throughput_mbps:>17.1f}")

        results.append(
            BenchResult(
                f"Length scaling L={L}",
                n_runs,
                mean_s * n_runs,
                f"{throughput_mbps:.1f} Mbit/s",
            )
        )

    return results


# ---------------------------------------------------------------------------
# Memory footprint tracking
# ---------------------------------------------------------------------------


def bench_memory_footprint() -> List[BenchResult]:
    """Peak memory allocation via tracemalloc for varying network sizes."""
    from sc_neurocore.layers.vectorized_layer import VectorizedSCLayer

    configs = [
        (32, 16, "tiny"),
        (64, 32, "small"),
        (128, 64, "medium"),
        (256, 128, "large"),
    ]
    L = 1024
    results: List[BenchResult] = []

    print(
        f"\n{'Config':<20} | {'Weight Matrix (MB)':>18} | {'Peak Alloc (MB)':>16} | {'Forward Time (ms)':>18}"
    )
    print("-" * 80)

    for n_inputs, n_neurons, label in configs:
        weight_bytes = n_neurons * n_inputs * 8  # float64

        tracemalloc.start()
        tracemalloc.reset_peak()

        layer = VectorizedSCLayer(n_inputs=n_inputs, n_neurons=n_neurons, length=L)
        x = np.random.uniform(0, 1, n_inputs).tolist()

        t0 = time.perf_counter()
        layer.forward(x)
        fwd_ms = (time.perf_counter() - t0) * 1e3

        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        weight_mb = weight_bytes / (1024 * 1024)
        peak_mb = peak_bytes / (1024 * 1024)

        tag = f"  {n_inputs}x{n_neurons} ({label})"
        print(f"{tag:<20}| {weight_mb:>16.3f}  | {peak_mb:>14.2f}  | {fwd_ms:>16.2f}")

        results.append(
            BenchResult(
                f"Memory {n_inputs}x{n_neurons} ({label})",
                1,
                fwd_ms / 1e3,
                f"peak={peak_mb:.2f} MB, weights={weight_mb:.3f} MB",
            )
        )

    return results


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def run_benchmarks(full: bool = False) -> List[BenchResult]:
    results: List[BenchResult] = []

    # Scale factors
    s = 10 if full else 1

    print("=" * 80)
    print("  SC-NeuroCore Benchmark Suite")
    print(f"  Backend: {_gpu_banner_label()}")
    print(f"  Mode: {'full' if full else 'quick'}")
    print("=" * 80)

    # 1. Scalar primitives
    print("\n--- Scalar Primitives ---")
    for b in [
        bench_lfsr(100_000 * s),
        bench_encoder(100_000 * s),
        bench_lif(100_000 * s),
    ]:
        results.append(b)
        print(b)

    # 2. Packed bitstream ops
    print("\n--- Packed Bitstream Operations (NumPy) ---")
    for b in [
        bench_pack_1d(1024, 1000 * s),
        bench_pack_1d(65536, 200 * s),
        bench_pack_2d(64, 1024, 200 * s),
        bench_vec_and(1024, 5000 * s),
        bench_popcount(1024, 5000 * s),
    ]:
        results.append(b)
        print(b)

    # 3. Dense layer
    print("\n--- Dense Layer Forward ---")
    for b in [
        bench_dense_forward(16, 8, 256, 50 * s),
        bench_dense_forward(64, 32, 1024, 10 * s),
    ]:
        results.append(b)
        print(b)

    # 4. Full pipeline
    print("\n--- Full Pipeline (encode->synapse->neuron) ---")
    for b in [
        bench_pipeline(4, 256, 20 * s),
        bench_pipeline(16, 256, 5 * s),
    ]:
        results.append(b)
        print(b)

    # 5. GPU / dual-path
    print(f"\n--- GPU Backend ({_gpu_backend_label()}) ---")
    for b in [
        bench_gpu_pack(65536, 200 * s),
        bench_gpu_mac(64, 32, 16, 100 * s),
    ]:
        results.append(b)
        print(b)

    # 6. Bitstream length scaling
    print("\n--- Bitstream Length Scaling (32x16 dense) ---")
    n_runs = 20 if full else 5
    results.extend(bench_bitstream_length_scaling(n_runs=n_runs))

    # 7. Memory footprint
    print("\n--- Memory Footprint (L=1024) ---")
    results.extend(bench_memory_footprint())

    return results


def write_markdown(results: List[BenchResult], path: str = "BENCHMARKS.md") -> None:
    lines = [
        "# SC-NeuroCore Performance Benchmarks",
        "",
        f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Backend: {'CuPy (CUDA)' if _gpu_runtime_available() else 'NumPy (CPU only)'}",
        "",
        "| Benchmark | Backend | Iterations | Avg Latency | Throughput |",
        "|-----------|---------|------------|-------------|------------|",
    ]
    for r in results:
        lines.append(r.md_row())
    lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nWrote benchmark results to {path}")


def write_json(results: List[BenchResult], path: str = "benchmark_results.json") -> None:
    import json

    entries = [
        {
            "name": r.name,
            "unit": "us/iter",
            "value": round(r.avg_us, 2),
            "extra": r.throughput,
        }
        for r in results
    ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)
    print(f"\nWrote JSON results to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="SC-NeuroCore Benchmark Suite")
    parser.add_argument(
        "--full", action="store_true", help="Run thorough benchmarks (10x iterations)"
    )
    parser.add_argument("--markdown", action="store_true", help="Write results to BENCHMARKS.md")
    parser.add_argument("--json", action="store_true", help="Write results as JSON for CI")
    args = parser.parse_args()

    results = run_benchmarks(full=args.full)

    print(f"\n{'=' * 80}")
    print(f"  {len(results)} benchmarks complete.")
    print(f"{'=' * 80}")

    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)

    if args.markdown:
        write_markdown(results, os.path.join(out_dir, "BENCHMARKS.md"))

    if args.json:
        write_json(results, os.path.join(out_dir, "benchmark_results.json"))


if __name__ == "__main__":
    main()
