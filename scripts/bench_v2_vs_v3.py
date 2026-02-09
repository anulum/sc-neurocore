#!/usr/bin/env python
"""
SC-NeuroCore v2 vs v3 Head-to-Head Benchmark Suite
===================================================

Measures wall-clock time for every operation that has both
a Python (v2) and Rust (v3) implementation.

Usage:
    cd 03_CODE/sc-neurocore
    $env:PYTHONPATH='src'
    .\\.venv\\Scripts\\python scripts/bench_v2_vs_v3.py
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

# -- v2 imports --
from sc_neurocore.accel.vector_ops import (
    pack_bitstream as v2_pack,
    unpack_bitstream as v2_unpack,
    vec_and as v2_and,
    vec_popcount as v2_popcount,
)
from sc_neurocore.layers import VectorizedSCLayer as V2Layer
from sc_neurocore.layers.attention import StochasticAttention as V2Attn
from sc_neurocore.neurons import FixedPointLIFNeuron as V2Lif

# -- v3 imports --
import sc_neurocore_engine as v3
from sc_neurocore_engine import FixedPointLIFNeuron as V3Lif
from sc_neurocore_engine import KuramotoSolver
from sc_neurocore_engine.attention import StochasticAttention as V3Attn
from sc_neurocore_engine.layers import VectorizedSCLayer as V3Layer


@dataclass
class BenchResult:
    name: str
    v2_ms: float
    v3_ms: float

    @property
    def speedup(self) -> float:
        return self.v2_ms / self.v3_ms if self.v3_ms > 0 else float("inf")


def bench(
    name: str,
    v2_fn: Callable,
    v3_fn: Callable,
    warmup: int = 3,
    repeats: int = 10,
) -> BenchResult:
    """Benchmark v2 and v3 implementations."""
    for _ in range(warmup):
        v2_fn()
        v3_fn()

    t0 = time.perf_counter()
    for _ in range(repeats):
        v2_fn()
    v2_ms = (time.perf_counter() - t0) * 1000 / repeats

    t0 = time.perf_counter()
    for _ in range(repeats):
        v3_fn()
    v3_ms = (time.perf_counter() - t0) * 1000 / repeats

    return BenchResult(name, v2_ms, v3_ms)


def main() -> int:
    rng = np.random.RandomState(42)
    results: list[BenchResult] = []

    print("SC-NeuroCore v2 vs v3 Benchmark Suite")
    print(f"SIMD tier: {v3.simd_tier()}")
    print("=" * 60)

    # Keep imports used to preserve parity with v2 vector_ops API surface.
    _ = (v2_unpack, v2_and)

    # -- 1. Pack Bitstream --
    bits_1m = rng.randint(0, 2, 1_000_000).astype(np.uint8)
    results.append(
        bench(
            "pack_bitstream (1M bits)",
            lambda: v2_pack(bits_1m),
            lambda: v3.pack_bitstream(bits_1m),
        )
    )

    # -- 2. Popcount --
    packed_1m = v2_pack(bits_1m)
    v3_packed = v3.pack_bitstream(bits_1m)
    results.append(
        bench(
            "popcount (1M bits)",
            lambda: v2_popcount(packed_1m),
            lambda: v3.popcount(v3_packed),
        )
    )

    # -- 3. LIF Neuron (10K steps) --
    def v2_lif_10k():
        lif = V2Lif()
        for _ in range(10_000):
            lif.step(20, 256, 128, 0)

    def v3_lif_10k():
        lif = V3Lif()
        for _ in range(10_000):
            lif.step(20, 256, 128, 0)

    results.append(bench("LIF neuron (10K steps)", v2_lif_10k, v3_lif_10k))

    # -- 4. Dense Layer Forward --
    for n_in, n_out in [(16, 8), (64, 32), (128, 64)]:
        length = 1024
        v2_layer = V2Layer(n_inputs=n_in, n_neurons=n_out, length=length, use_gpu=False)
        v3_layer = V3Layer(n_inputs=n_in, n_neurons=n_out, length=length)
        inp = rng.uniform(0.1, 0.9, n_in)

        results.append(
            bench(
                f"Dense forward ({n_in}->{n_out}, L={length})",
                lambda i=inp, ly=v2_layer: ly.forward(i),
                lambda i=inp, ly=v3_layer: ly.forward(i),
            )
        )

    # -- 5. Attention --
    for n, m, dk, dv in [(10, 20, 16, 32), (50, 100, 32, 64)]:
        q = rng.uniform(0, 1, (n, dk))
        k = rng.uniform(0, 1, (m, dk))
        v = rng.uniform(0, 1, (m, dv))

        v2_attn = V2Attn(dim_k=dk)
        v3_attn = V3Attn(dim_k=dk)

        results.append(
            bench(
                f"Attention ({n}x{dk} -> {m}x{dv})",
                lambda q=q, k=k, v=v, a=v2_attn: a.forward(q, k, v),
                lambda q=q, k=k, v=v, a=v3_attn: a.forward(q, k, v),
            )
        )

    # -- 6. Kuramoto Solver (v3 only — no direct v2 equivalent) --
    n_osc = 400
    omega = np.ones(n_osc)
    coupling = rng.uniform(0, 0.5, (n_osc, n_osc))
    coupling = (coupling + coupling.T) / 2
    phases = rng.uniform(0, 2 * np.pi, n_osc)

    def v3_kuramoto_1000():
        solver = KuramotoSolver(omega, coupling, phases, noise_amp=0.0)
        solver.run(1000, 0.01)

    t0 = time.perf_counter()
    for _ in range(3):
        v3_kuramoto_1000()
    v3_k_ms = (time.perf_counter() - t0) * 1000 / 3

    print()
    print(f"{'Operation':<45} {'v2 (ms)':>10} {'v3 (ms)':>10} {'Speedup':>10}")
    print("-" * 77)
    for result in results:
        print(
            f"{result.name:<45} "
            f"{result.v2_ms:>10.3f} {result.v3_ms:>10.3f} {result.speedup:>9.1f}x"
        )
    print("-" * 77)
    print(
        f"{'Kuramoto 400 osc x 1000 steps (v3 only)':<45} "
        f"{'N/A':>10} {v3_k_ms:>10.3f} {'-':>10}"
    )
    print()

    speedups = [result.speedup for result in results if result.speedup < float("inf")]
    if speedups:
        geo_mean = np.exp(np.mean(np.log(speedups)))
        print(f"Geometric mean speedup: {geo_mean:.1f}x")

    return 0


if __name__ == "__main__":
    sys.exit(main())
