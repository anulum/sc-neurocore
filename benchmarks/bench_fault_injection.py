#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — FaultInjector wall-clock benchmark across 5 fault models

"""Reproducible benchmark for `sc_neurocore.fault_injection.FaultInjector.inject()`.

For each of the 5 fault models (BIT_FLIP / STUCK_AT_0 / STUCK_AT_1 /
GAUSSIAN_NOISE / DROPOUT), measure wall time for a single
`inject(bitstream, model, ber)` call on a 1 Mbit boolean array.

Median + min over 5 repeats reported per fault model.

Usage:
    python benchmarks/bench_fault_injection.py
    python benchmarks/bench_fault_injection.py --json benchmarks/results/bench_fault_injection.json
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

from sc_neurocore.fault_injection import (
    FaultInjector,
    FaultModel,
)


N_REPEATS = 5
N_BITS = 1_000_000  # 1 Mbit — representative SC bitstream

# Use a higher-than-LEO BER so fault counts are non-zero on a 1 M-bit
# stream within the inner loop's RNG sample. LEO BER (1e-7) gives
# ~0.1 expected faults per call — too noisy for benchmarking.
BENCH_BER = 1e-3


def bench_one(model: FaultModel, ber: float) -> tuple[float, float]:
    """Return (median_ms, min_ms) over N_REPEATS for inject()."""
    rng = np.random.default_rng(42)
    bitstream = rng.integers(0, 2, size=N_BITS, dtype=np.uint8).astype(bool)
    injector = FaultInjector(seed=42)

    # Warm-up
    injector.inject(bitstream.copy(), model, ber)

    times_ms: list[float] = []
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        injector.inject(bitstream.copy(), model, ber)
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    times_ms.sort()
    return times_ms[len(times_ms) // 2], times_ms[0]


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="FaultInjector.inject() wall-clock benchmark across 5 fault models."
    )
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    print(f"# FaultInjector.inject() benchmark")
    print(f"# Bitstream length: {N_BITS}, BER: {BENCH_BER:.0e}")
    print(f"# (BER raised from LEO 1e-7 so fault counts are non-zero per call)")
    print(f"# Repeats per cell: {N_REPEATS}")
    print(f"# Python: {platform.python_version()}, NumPy: {np.__version__}")
    print(f"# platform: {platform.platform()}")
    print()
    print(f"{'fault model':<20}  {'median ms':>12}  {'min ms':>12}")
    print(f"{'-'*20}  {'-'*12}  {'-'*12}")

    rows: list[dict[str, object]] = []
    for model in FaultModel:
        median_ms, min_ms = bench_one(model, BENCH_BER)
        print(f"{model.name:<20}  {median_ms:>12.3f}  {min_ms:>12.3f}")
        rows.append({
            "fault_model": model.name,
            "n_bits": N_BITS,
            "ber": BENCH_BER,
            "median_ms": median_ms,
            "min_ms": min_ms,
        })

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "n_bits": N_BITS,
            "n_repeats": N_REPEATS,
            "ber": BENCH_BER,
            "rows": rows,
        }
        args.json.write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
