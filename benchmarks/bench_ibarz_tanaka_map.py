#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Ibarz-Tanaka piecewise-linear map multi-language benchmark

"""Multi-language benchmark for ``IbarzTanakaMapNeuron.simulate``.

Times the N-step piecewise-linear map recurrence across the polyglot backend
chain (python / rust / julia / go / mojo), records the parity gap against the
NumPy reference, and writes a JSON artefact.

Usage::

    python benchmarks/bench_ibarz_tanaka_map.py
    python benchmarks/bench_ibarz_tanaka_map.py --json benchmarks/results/bench_ibarz_tanaka_map.json

Measurement note: functional / local-regression benchmark on a loaded
workstation, explicitly **non-isolated** per
`BROADCAST_2026-06-04_benchmark_core_isolation`; do not promote the speed
numbers without an isolated-core rerun.
"""

from __future__ import annotations

import argparse
import json
import os as _os
import platform
import time
from pathlib import Path

import numpy as np

from sc_neurocore.neurons.models import ibarz_tanaka_map
from sc_neurocore.neurons.models.ibarz_tanaka_map import IbarzTanakaMapNeuron

N_STEPS = 2_000_000
CURRENT = 3.0  # drives the spiking branch (defaults are silent below ~2.0)
N_REPEATS = 5


def _probe_rust() -> tuple[bool, str]:
    return (
        ibarz_tanaka_map._HAS_RUST,
        "" if ibarz_tanaka_map._HAS_RUST else "engine wheel lacks the symbol",
    )


def _probe_julia() -> tuple[bool, str]:
    ok = ibarz_tanaka_map._ensure_julia_loaded()
    return (ok, "" if ok else "juliacall/.jl unavailable")


def _probe_go() -> tuple[bool, str]:
    ok = ibarz_tanaka_map._ensure_go_loaded()
    return (ok, "" if ok else "accel/go/neurons/ibarz_tanaka_map/libibarz.so not built")


def _probe_mojo() -> tuple[bool, str]:
    if not _os.path.isfile(_os.path.expanduser("~/.pixi/bin/mojo")):
        return False, "mojo binary not at ~/.pixi/bin/mojo"
    ok = ibarz_tanaka_map._ensure_mojo_loaded()
    return (ok, "" if ok else "accel/mojo/neurons/libibarz.so not built")


def _run(backend: str) -> tuple[float, float, np.ndarray]:
    IbarzTanakaMapNeuron().simulate(N_STEPS, CURRENT, backend=backend)  # warm-up (JIT for Julia)
    times_ms: list[float] = []
    trace = np.empty(0)
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        trace, _spikes = IbarzTanakaMapNeuron().simulate(N_STEPS, CURRENT, backend=backend)
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    times_ms.sort()
    return times_ms[len(times_ms) // 2], times_ms[0], trace


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Ibarz-Tanaka piecewise-linear map multi-language benchmark."
    )
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    print("# Ibarz-Tanaka piecewise-linear map N-step benchmark")
    print(f"# Workload: {N_STEPS:,} steps, default params, current={CURRENT}")
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

    reference: np.ndarray | None = None
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
            assert reference is not None
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
    print("# Note: rust/julia/go reproduce the trace bit-for-bit (parity 0).")
    print("# Mojo's release build can contract the linear-branch and slow-variable")
    print("# multiply-adds into FMAs; the per-spike reset resynchronises the")
    print("# trajectory so the whole-trace gap stays at the per-step ULP level.")

    report = {
        "benchmark": "ibarz_tanaka_map_simulate",
        "workload": {"n_steps": N_STEPS, "current": CURRENT, "repeats": N_REPEATS},
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
