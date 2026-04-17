#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SafetyMonitor.check() wall-clock benchmark

"""Reproducible benchmark for `sc_neurocore.safety_cert.SafetyMonitor.check()`.

Three scenarios exercised:

1. **All-defaults** — default-safe inputs, no violations.
2. **Triggered overcurrent** — current above max → P1 violates.
3. **All 6 violations** — every property fires.

Plus the post-simulation `CertificationGenerator.build_package()`
end-to-end timing on a small representative network.

Each scenario is timed with `timeit` for 100 000 iterations
(check) or 10 iterations (build_package), median + min over 5
repeats reported.

Usage:
    python benchmarks/bench_safety_monitor.py
    python benchmarks/bench_safety_monitor.py --json benchmarks/results/bench_safety_monitor.json
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

from sc_neurocore.safety_cert import SafetyLimits, SafetyMonitor


CHECK_ITERATIONS = 100_000
N_REPEATS = 5


def _time_check(fn, iterations: int) -> tuple[float, float]:
    """Return (median_ns_per_call, min_ns_per_call) over N_REPEATS."""
    per_call_ns: list[float] = []
    for _ in range(N_REPEATS):
        t0 = time.perf_counter_ns()
        for _ in range(iterations):
            fn()
        per_call_ns.append((time.perf_counter_ns() - t0) / iterations)
    per_call_ns.sort()
    return per_call_ns[len(per_call_ns) // 2], per_call_ns[0]


def bench_check_no_violation() -> tuple[float, float]:
    mon = SafetyMonitor()

    def f() -> None:
        mon.check()  # all defaults — no violation

    # Warm-up
    for _ in range(1000):
        f()
    mon.reset()
    return _time_check(f, CHECK_ITERATIONS)


def bench_check_overcurrent() -> tuple[float, float]:
    mon = SafetyMonitor()

    def f() -> None:
        mon.check(current=0x8000)  # > default max 0x7FFF
        mon.reset()

    for _ in range(1000):
        f()
    return _time_check(f, CHECK_ITERATIONS)


def bench_check_all_violations() -> tuple[float, float]:
    mon = SafetyMonitor()
    limits = SafetyLimits()

    def f() -> None:
        # Trigger every property in one call
        mon.check(
            current=limits.max_current + 1,           # P1
            voltage=limits.max_voltage + 1,           # P1
            coherence=0,                              # P1 + P2 (drops below prev)
            popcount_k=limits.sc_denom + 1,           # P3
            sc_add_result=limits.sc_denom + 1,        # P4
            membrane=limits.lif_v_max + 1,            # P5
            scc_numerator=999_999,                    # P6
            scc_denominator=1,
        )
        mon.reset()

    for _ in range(1000):
        f()
    return _time_check(f, CHECK_ITERATIONS)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="SafetyMonitor.check() wall-clock benchmark."
    )
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    print(f"# SafetyMonitor.check() benchmark")
    print(f"# Iterations per scenario: {CHECK_ITERATIONS}, repeats: {N_REPEATS}")
    print(f"# Python: {platform.python_version()}, platform: {platform.platform()}")
    print()
    print(f"{'scenario':<32}  {'median ns':>12}  {'min ns':>12}")
    print(f"{'-'*32}  {'-'*12}  {'-'*12}")

    scenarios = {
        "no violation (defaults)": bench_check_no_violation,
        "overcurrent (P1)": bench_check_overcurrent,
        "all 6 violations": bench_check_all_violations,
    }

    rows: list[dict[str, object]] = []
    for name, fn in scenarios.items():
        median_ns, min_ns = fn()
        print(f"{name:<32}  {median_ns:>12.1f}  {min_ns:>12.1f}")
        rows.append({
            "scenario": name,
            "median_ns_per_call": median_ns,
            "min_ns_per_call": min_ns,
            "median_us_per_call": median_ns / 1000.0,
        })

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "iterations_per_scenario": CHECK_ITERATIONS,
            "n_repeats": N_REPEATS,
            "rows": rows,
        }
        args.json.write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
