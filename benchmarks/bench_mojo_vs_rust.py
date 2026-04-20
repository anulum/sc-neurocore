# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo vs Python Benchmark Comparison

"""Runs the Mojo SIMD kernel suite and compares against pure-Python equivalents.

Both sides time the *same inner kernel* over their own loop count, and the
comparison normalises to **per-call time (ns)** so the Mojo loop count
(100k–1M) and the Python loop count (1 000) are directly comparable.
"""

import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "..", "..", "src"))


# Mojo-side label → (normalised label, Mojo loop count).
# The Mojo kernels.mojo file labels with "§N  Name NNw × NM / N × 1k" etc.;
# we map those onto a canonical name + total iteration count for apples-to-
# apples ns/call comparison.
MOJO_LABEL_MAP = {
    "§1  Popcount 1024w × 1M": ("popcount_1024w", 1_000_000),
    "§2  SCC 256w × 1M": ("scc_numerator_256w", 1_000_000),
    "§3  LFSR 1024-bit × 100k": ("lfsr_encode_1024bit", 100_000),
    "§7  STDP 1024w × 100k": ("stdp_update_1024w", 100_000),
    "§8  HDC similarity 256w × 1M": ("hdc_bind_256w", 1_000_000),
    "§14 Attention score 256w × 1M": ("attention_256w", 1_000_000),
    "§18 Histogram 1024w/32 × 10k": ("histogram_1024w", 10_000),
    "§19 LIF batch 64 × 100k": ("lif_batch_64", 100_000),
    "§23 Sobol 1024-bit × 100k": ("sobol_1024bit", 100_000),
    "§31 Spike bin 10k × 10k": ("spike_bin_10k", 10_000),
    "§33 DVS pack 4k × 10k": ("dvs_pack_4k", 10_000),
    "§35 Ring topo 64 × 1k": ("ring_topo_64", 1_000),
    "§37 DNA hamming 256w × 1M": ("dna_hamming_256w", 1_000_000),
}


def run_mojo_bench():
    """Execute kernels.mojo via pixi and return {canonical_label: ns_per_call}."""
    from sc_neurocore.accel.mojo import MojoKernelRunner

    runner = MojoKernelRunner()
    raw = runner.run_benchmark(timeout_sec=120)
    normalised: dict[str, float] = {}
    for mojo_label, total_ms in raw.items():
        key = mojo_label.strip()
        if key in MOJO_LABEL_MAP:
            canonical, iters = MOJO_LABEL_MAP[key]
            ns_per_call = (total_ms * 1_000_000.0) / iters
            normalised[canonical] = ns_per_call
    return normalised


def run_python_bench():
    """Pure-Python equivalents, reported as ns per single call."""
    from sc_neurocore.edge.bitstream import popcount_slice, popcount32
    from sc_neurocore.edge.lfsr import Lfsr16

    results: dict[str, float] = {}

    # popcount_1024w
    data = [0xDEADBEEF] * 1024
    iters = 1_000
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = popcount_slice(data)
    t1 = time.perf_counter()
    results["popcount_1024w"] = (t1 - t0) * 1e9 / iters

    # scc_numerator_256w
    a = [0xAAAAAAAA] * 256
    b = [0x55555555] * 256
    iters = 1_000
    t0 = time.perf_counter()
    for _ in range(iters):
        pa = sum(popcount32(x) for x in a)
        pb = sum(popcount32(x) for x in b)
        pab = sum(popcount32(x & y) for x, y in zip(a, b))
        _ = pab * len(a) * 32 - pa * pb
    t1 = time.perf_counter()
    results["scc_numerator_256w"] = (t1 - t0) * 1e9 / iters

    # lfsr_encode_1024bit
    lfsr = Lfsr16(0xACE1)
    iters = 1_000
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = lfsr.encode(32768, 1024)
    t1 = time.perf_counter()
    results["lfsr_encode_1024bit"] = (t1 - t0) * 1e9 / iters

    return results


def main():
    print("=" * 80)
    print("SC-NeuroCore: Mojo SIMD vs Pure Python Benchmark (normalised ns/call)")
    print("=" * 80)

    print("\n[1/2] Running Mojo benchmarks (runs the full kernel suite once)...")
    mojo = run_mojo_bench()
    if not mojo:
        print("  Mojo unavailable, showing Python-only results.")

    print("[2/2] Running Python baseline benchmarks...")
    py = run_python_bench()

    cols = f"\n{'Benchmark':<25} {'Mojo (ns)':>14} {'Python (ns)':>14} {'Speedup':>10}"
    print(cols)
    print("-" * 80)
    union = sorted(set(py) | set(mojo))
    for key in union:
        py_ns = py.get(key)
        mojo_ns = mojo.get(key)
        if py_ns and mojo_ns and mojo_ns > 0:
            speedup = py_ns / mojo_ns
            print(f"{key:<25} {mojo_ns:>14.1f} {py_ns:>14.1f} {speedup:>9.1f}×")
        elif py_ns and not mojo_ns:
            print(f"{key:<25} {'N/A':>14} {py_ns:>14.1f} {'—':>10}")
        elif mojo_ns and not py_ns:
            print(f"{key:<25} {mojo_ns:>14.1f} {'—':>14} {'—':>10}")

    print("=" * 80)


if __name__ == "__main__":
    main()
