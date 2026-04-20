# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo vs Python Benchmark Comparison

"""Runs the Mojo SIMD kernel suite and compares against pure-Python equivalents."""

import os
import re
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "..", "..", "src"))


def run_mojo_bench():
    """Execute Mojo kernels.mojo and parse all benchmark lines."""
    from sc_neurocore.accel.mojo import MojoKernelRunner
    
    runner = MojoKernelRunner()
    return runner.run_benchmark(timeout_sec=60)


def run_python_bench():
    """Run pure-Python equivalents of the Mojo benchmarks."""
    from sc_neurocore.edge.bitstream import popcount_slice, popcount32, MASK32
    from sc_neurocore.edge.lfsr import Lfsr16

    timings = {}

    data = [0xDEADBEEF] * 1024
    t0 = time.perf_counter()
    for _ in range(1000):
        _ = popcount_slice(data)
    t1 = time.perf_counter()
    timings["Popcount 1024×u32 × 1000 iters"] = (t1 - t0) * 1000

    a = [0xAAAAAAAA] * 256
    b = [0x55555555] * 256
    t0 = time.perf_counter()
    for _ in range(1000):
        pa = sum(popcount32(x) for x in a)
        pb = sum(popcount32(x) for x in b)
        pab = sum(popcount32(x & y) for x, y in zip(a, b))
        _ = pab * len(a) * 32 - pa * pb
    t1 = time.perf_counter()
    timings["SCC 256×u32 × 1000 iters"] = (t1 - t0) * 1000

    lfsr = Lfsr16(0xACE1)
    t0 = time.perf_counter()
    for _ in range(1000):
        _ = lfsr.encode(32768, 1024)
    t1 = time.perf_counter()
    timings["LFSR encode 1024-bit × 1000 iters"] = (t1 - t0) * 1000

    return timings


def main():
    print("=" * 65)
    print("SC-NeuroCore: Mojo SIMD vs Pure Python Benchmark")
    print("=" * 65)

    print("\n[1/2] Running Mojo benchmarks...")
    mojo = run_mojo_bench()
    if not mojo:
        print("  Mojo unavailable, showing Python-only results.")

    print("[2/2] Running Python benchmarks...")
    python = run_python_bench()

    print(f"\n{'Benchmark':<45} {'Mojo (ms)':>12} {'Python (ms)':>12} {'Speedup':>10}")
    print("-" * 80)
    for label in python:
        py_val = python[label]
        mojo_val = mojo.get(label)
        if mojo_val and mojo_val > 0:
            speedup = py_val / mojo_val
            print(f"{label:<45} {mojo_val:>12.4f} {py_val:>12.4f} {speedup:>9.1f}×")
        else:
            print(f"{label:<45} {'N/A':>12} {py_val:>12.4f} {'—':>10}")

    print("=" * 65)


if __name__ == "__main__":
    main()
