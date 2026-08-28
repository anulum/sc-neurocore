#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mihalas-Niebur 2009 generalized IF multi-language benchmark

"""Multi-language benchmark for ``MihalasNieburNeuron.simulate``.

Times the N-step RK4 recurrence across the polyglot backend chain
(python / rust / julia / go / mojo), records the parity gap against the NumPy
reference, and writes a JSON artefact.

Usage::

    python benchmarks/bench_mihalas_niebur_simulate.py
    python benchmarks/bench_mihalas_niebur_simulate.py --json benchmarks/results/bench_mihalas_niebur_simulate.json

Measurement note: functional / local-regression benchmark on a loaded
workstation, explicitly **non-isolated** per
`BROADCAST_2026-06-04_benchmark_core_isolation`; do not promote the speed
numbers without an isolated-core rerun.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from sc_neurocore.neurons.models import mihalas_niebur
from sc_neurocore.neurons.models.mihalas_niebur import MihalasNieburNeuron

N_STEPS = 200_000
CURRENT = 0.002
N_REPEATS = 3
MODEL_KWARGS = {"current_jump_1": 0.01, "current_jump_2": -0.0006}


def _probe_rust() -> tuple[bool, str]:
    ok = mihalas_niebur._HAS_RUST
    return (ok, "" if ok else "engine wheel lacks the symbol")


def _probe_julia() -> tuple[bool, str]:
    ok = mihalas_niebur._ensure_julia_loaded()
    return (ok, "" if ok else "juliacall/.jl unavailable")


def _probe_go() -> tuple[bool, str]:
    ok = mihalas_niebur._ensure_go_loaded()
    return (ok, "" if ok else "accel/go/neurons/mihalas_niebur/libmihalasniebur.so not built")


def _probe_mojo() -> tuple[bool, str]:
    ok = mihalas_niebur._ensure_mojo_loaded()
    return (ok, "" if ok else "accel/mojo/neurons/libmihalasniebur.so not built")


def _run(backend: str) -> tuple[float, float, NDArray[np.float64]]:
    MihalasNieburNeuron(**MODEL_KWARGS).simulate(
        N_STEPS, CURRENT, backend=backend
    )  # warm-up (Julia JIT)
    times_ms: list[float] = []
    trace: NDArray[np.float64] = np.empty((0, 4), dtype=np.float64)
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        trace, _spikes = MihalasNieburNeuron(**MODEL_KWARGS).simulate(
            N_STEPS, CURRENT, backend=backend
        )
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    times_ms.sort()
    return times_ms[len(times_ms) // 2], times_ms[0], trace


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Mihalas-Niebur 2009 generalized IF multi-language benchmark."
    )
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    print("# Mihalas-Niebur 2009 generalized IF N-step RK4 benchmark")
    print(f"# Workload: {N_STEPS:,} source-profile panel-M steps, current={CURRENT}")
    print(f"# Repeats per backend: {N_REPEATS}")
    print(f"# Python: {platform.python_version()}, NumPy: {np.__version__}")
    print(f"# platform: {platform.platform()}")
    print("# isolation: non-isolated (loaded workstation) — functional/regression evidence")
    print()

    backends = {
        "python": (True, ""),
        "rust": _probe_rust(),
        "julia": _probe_julia(),
        "go": _probe_go(),
        "mojo": _probe_mojo(),
    }

    print(f"{'backend':<8}  {'available':<10}  reason")
    print(f"{'-' * 8}  {'-' * 10}  {'-' * 50}")
    for name, (avail, reason) in backends.items():
        print(f"{name:<8}  {'yes' if avail else 'no':<10}  {reason}")
    print()

    reference: NDArray[np.float64] | None = None
    python_median: float | None = None
    rows: list[dict[str, object]] = []

    print(f"{'backend':<8}  {'median ms':>12}  {'min ms':>12}  {'parity Δ':>12}  {'speedup':>9}")
    print(f"{'-' * 8}  {'-' * 12}  {'-' * 12}  {'-' * 12}  {'-' * 9}")
    for name, (avail, reason) in backends.items():
        if not avail:
            print(f"{name:<8}  {'(skip)':>12}  {'(skip)':>12}  {'-':>12}  {'-':>9}")
            rows.append({"backend": name, "skipped": True, "unavailable_reason": reason})
            continue
        median_ms, min_ms, trace = _run(name)
        if name == "python":
            reference = trace
            python_median = median_ms
            parity = 0.0
        else:
            if reference is None:
                raise RuntimeError("Python reference must run before optional backends")
            parity = float(np.max(np.abs(trace - reference)))
        speedup = (python_median / median_ms) if python_median and median_ms > 0 else float("nan")
        print(f"{name:<8}  {median_ms:>12.2f}  {min_ms:>12.2f}  {parity:>12.2e}  {speedup:>8.2f}x")
        rows.append(
            {
                "backend": name,
                "median_ms": median_ms,
                "min_ms": min_ms,
                "parity_max_abs_diff": parity,
                "speedup_vs_python": speedup,
            }
        )

    print()
    print("# Note: the Mihalas-Niebur right-hand side is purely linear (no transcendental")
    print("# functions), so rust, julia and go reproduce the trace bit-for-bit (parity 0).")
    print("# Mojo may fuse multiply-add and is validated within the measured ULP-scale band.")

    report = {
        "benchmark": "mihalas_niebur_simulate",
        "workload": {
            "n_steps": N_STEPS,
            "current": CURRENT,
            "current_jump_1": MODEL_KWARGS["current_jump_1"],
            "current_jump_2": MODEL_KWARGS["current_jump_2"],
            "repeats": N_REPEATS,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "isolation": "non-isolated (loaded workstation)",
        },
        "results": rows,
    }
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"\nWrote {args.json}")
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))
