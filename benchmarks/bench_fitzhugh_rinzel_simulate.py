#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — FitzHugh-Rinzel RK4 simulator multi-language benchmark

"""Multi-language benchmark for ``FitzHughRinzelNeuron.simulate`` (RK4).

Times the N-step RK4 recurrence across the polyglot backend chain
(python / rust / julia / go / mojo), records the parity gap against the NumPy
reference, and writes a JSON artefact.

Usage::

    python benchmarks/bench_fitzhugh_rinzel_simulate.py
    python benchmarks/bench_fitzhugh_rinzel_simulate.py --json benchmarks/results/bench_fitzhugh_rinzel_simulate.json

Measurement note: functional / local-regression benchmark on a loaded
workstation, explicitly **non-isolated** per
`BROADCAST_2026-06-04_benchmark_core_isolation`; do not promote the speed
numbers without an isolated-core rerun.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import numpy.typing as npt

from sc_neurocore.neurons.models import fitzhugh_rinzel as fhr
from sc_neurocore.neurons.models.fitzhugh_rinzel import FitzHughRinzelNeuron

N_STEPS = 2_000_000
CURRENT = 0.5
N_REPEATS = 5
ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    "benchmarks/bench_fitzhugh_rinzel_simulate.py",
    "engine/src/bindings/fitzhugh_rinzel.rs",
    "engine/src/neurons/simple_spiking/fitzhugh_rinzel.rs",
    "src/sc_neurocore/accel/go/neurons/fitzhugh_rinzel/fitzhugh_rinzel.go",
    "src/sc_neurocore/accel/julia/neurons/fitzhugh_rinzel.jl",
    "src/sc_neurocore/accel/mojo/neurons/fitzhugh_rinzel.mojo",
    "src/sc_neurocore/accel/rust/safety/fitzhugh_rinzel.rs",
    "src/sc_neurocore/neurons/model_descriptors/FitzHughRinzelNeuron.toml",
    "src/sc_neurocore/neurons/model_schemas/fitzhugh_rinzel.json",
    "src/sc_neurocore/neurons/model_schemas/fitzhugh_rinzel.toml",
    "src/sc_neurocore/neurons/models/fitzhugh_rinzel.py",
    "src/sc_neurocore/neurons/reference_trace_data/fitzhugh_rinzel_driven_bursting_doi.json",
)


def _source_hashes() -> dict[str, object]:
    """Return flat digests plus suffix aliases consumed by the evidence gate."""
    hashes: dict[str, object] = {}
    for relative in SOURCES:
        digest = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        hashes[relative] = digest
        stem, suffix = relative.rsplit(".", 1)
        aliases = hashes.setdefault(stem, {})
        if not isinstance(aliases, dict):
            raise RuntimeError(f"source-hash alias collision at {stem}")
        aliases[suffix] = digest
    return hashes


def _probe_rust() -> tuple[bool, str]:
    return (fhr._HAS_RUST, "" if fhr._HAS_RUST else "engine wheel lacks the symbol")


def _probe_julia() -> tuple[bool, str]:
    ok = fhr._ensure_julia_loaded()
    return (ok, "" if ok else "juliacall/.jl unavailable")


def _probe_go() -> tuple[bool, str]:
    ok = fhr._ensure_go_loaded()
    return (ok, "" if ok else "accel/go/neurons/fitzhugh_rinzel/libfhr.so not built")


def _probe_mojo() -> tuple[bool, str]:
    ok = fhr._ensure_mojo_loaded()
    return (ok, "" if ok else "accel/mojo/neurons/libfhr.so not built")


def _run(backend: str) -> tuple[float, float, npt.NDArray[np.float64]]:
    FitzHughRinzelNeuron().simulate(N_STEPS, CURRENT, backend=backend)  # warm-up (Julia JIT)
    times_ms: list[float] = []
    trace: npt.NDArray[np.float64] = np.empty(0, dtype=np.float64)
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        trace, _spikes = FitzHughRinzelNeuron().simulate(N_STEPS, CURRENT, backend=backend)
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    times_ms.sort()
    return times_ms[len(times_ms) // 2], times_ms[0], trace


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="FitzHugh-Rinzel RK4 multi-language benchmark.")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    print("# FitzHugh-Rinzel RK4 N-step benchmark")
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
    unavailable = {name: reason for name, (available, reason) in backends.items() if not available}
    if unavailable:
        print(f"Required backend unavailable; evidence was not written: {unavailable}")
        return 2

    reference: npt.NDArray[np.float64] | None = None
    python_median: float | None = None
    rows: dict[str, dict[str, float]] = {}

    print(f"{'backend':<8}  {'median ms':>12}  {'min ms':>12}  {'parity Δ':>12}  {'speedup':>9}")
    print(f"{'-' * 8}  {'-' * 12}  {'-' * 12}  {'-' * 12}  {'-' * 9}")
    for name in backends:
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
        rows[name] = {
            "median_ms": median_ms,
            "min_ms": min_ms,
            "parity_max_abs_diff": parity,
            "speedup_vs_python": speedup,
        }

    print()
    print("# Note: the RHS is exact arithmetic (v*v*v, no transcendentals), so")
    print("# rust/julia/go reproduce the trace bit-for-bit (parity 0). Mojo fuses")
    print("# some RK4 multiply-adds into FMAs; the slow mu=1e-4 recovery keeps the")
    print("# dynamics from being strongly chaotic, so the gap stays a non-amplifying")
    print("# ULP band with identical spike counts. auto -> Rust (wheel-shipped bit-exact backend).")

    report = {
        "schema_version": "sc-neurocore.polyglot-benchmark.v1",
        "benchmark": "fitzhugh_rinzel_simulate_rk4",
        "model": "FitzHughRinzelNeuron",
        "evidence_class": "local_regression_non_isolated",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "workload": {"n_steps": N_STEPS, "current": CURRENT, "repeats": N_REPEATS},
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "isolation": "non-isolated (loaded workstation)",
        },
        "backends": rows,
        "source_hashes": _source_hashes(),
        "production_speed_claim": False,
        "hardware_measurement_claimed": False,
    }
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"\nWrote {args.json}")
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))
