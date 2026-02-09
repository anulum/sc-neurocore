"""Head-to-head benchmark helper for SC-NeuroCore v2 vs v3."""

from __future__ import annotations

import time

import numpy as np

from sc_neurocore.accel.vector_ops import pack_bitstream as v2_pack
from sc_neurocore.accel.vector_ops import vec_popcount as v2_popcount

import sc_neurocore_engine as v3


def benchmark_pack_popcount(length: int = 1_000_000, repeats: int = 10) -> None:
    rng = np.random.RandomState(42)
    bits = rng.randint(0, 2, length).astype(np.uint8)

    t0 = time.perf_counter()
    for _ in range(repeats):
        packed = v2_pack(bits)
        v2_popcount(packed)
    t1 = time.perf_counter()

    t2 = time.perf_counter()
    for _ in range(repeats):
        packed = v3.pack_bitstream(bits)
        v3.popcount(packed)
    t3 = time.perf_counter()

    v2_ms = (t1 - t0) * 1000 / repeats
    v3_ms = (t3 - t2) * 1000 / repeats
    speedup = v2_ms / v3_ms if v3_ms > 0 else float("inf")

    print(f"Length: {length:,} bits")
    print(f"v2 avg: {v2_ms:.3f} ms")
    print(f"v3 avg: {v3_ms:.3f} ms")
    print(f"Speedup: {speedup:.2f}x")


if __name__ == "__main__":
    benchmark_pack_popcount()
