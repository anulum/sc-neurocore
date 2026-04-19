# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — C-FFI Bridge Benchmark (core_engine vs NumPy)

"""Benchmark Rust C-FFI core_engine vs pure NumPy for SC arithmetic."""

from __future__ import annotations

import json
import time
import numpy as np


def _numpy_popcount(packed: np.ndarray) -> int:
    x = packed.copy()
    x -= (x >> 1) & 0x5555555555555555
    x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333)
    x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0F
    x = (x * 0x0101010101010101) >> 56
    return int(np.sum(x))


def _bench(fn, data, label, warmup=3, rounds=10):
    for _ in range(warmup):
        fn(data)
    times = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        result = fn(data)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    mean_us = np.mean(times) * 1e6
    return {"label": label, "mean_us": round(mean_us, 2), "result": result}


def main():
    from sc_neurocore._native.core_engine_bridge import (
        is_available,
        sc_popcount_packed as rust_popcount,
    )

    if not is_available():
        print("ERROR: core_engine not available")
        return

    results = []

    for n_words in [64, 256, 1024, 4096, 16384]:
        data_np = np.random.randint(0, 2**63, size=n_words, dtype=np.uint64)
        data_list = data_np.tolist()

        r_np = _bench(_numpy_popcount, data_np, f"numpy_{n_words}w")
        r_rs = _bench(rust_popcount, data_list, f"rust_{n_words}w")

        speedup = r_np["mean_us"] / max(r_rs["mean_us"], 0.01)
        assert r_np["result"] == r_rs["result"], f"Mismatch at {n_words}w!"

        print(
            f"  {n_words:>6} words | NumPy: {r_np['mean_us']:>10.2f} µs | "
            f"Rust: {r_rs['mean_us']:>10.2f} µs | "
            f"Speedup: {speedup:>6.1f}×"
        )

        results.append(
            {
                "n_words": n_words,
                "numpy_us": r_np["mean_us"],
                "rust_us": r_rs["mean_us"],
                "speedup": round(speedup, 1),
            }
        )

    out_path = "benchmarks/results/cffi_bridge_benchmark.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    print("SC-NeuroCore C-FFI Bridge Benchmark")
    print("=" * 60)
    main()
