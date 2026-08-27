#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — FitzHugh-Nagumo RK4 simulator multi-language benchmark

"""Multi-language benchmark for ``FitzHughNagumoNeuron.simulate`` (RK4).

Times the N-step RK4 recurrence across the polyglot backend chain
(python / rust / julia / go / mojo), records the parity gap against the NumPy
reference, and writes a JSON artefact.

Usage::

    python benchmarks/bench_fitzhugh_nagumo_simulate.py
    python benchmarks/bench_fitzhugh_nagumo_simulate.py --json benchmarks/results/bench_fitzhugh_nagumo_simulate.json

Measurement note: functional / local-regression benchmark on a loaded
workstation, explicitly **non-isolated** per
`BROADCAST_2026-06-04_benchmark_core_isolation`; do not promote the speed
numbers without an isolated-core rerun.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os as _os
import platform
import time
from pathlib import Path
from typing import TypedDict

import numpy as np
import numpy.typing as npt

from sc_neurocore.neurons.models import fitzhugh_nagumo as fhn
from sc_neurocore.neurons.models.fitzhugh_nagumo import FitzHughNagumoNeuron

N_STEPS = 2_000_000
CURRENT = 0.5
N_REPEATS = 5
ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "benchmarks/results/bench_fitzhugh_nagumo_simulate.json"
BACKENDS = ("python", "rust", "julia", "go", "mojo")
PARITY_ATOL = {
    "python": 0.0,
    "rust": 0.0,
    "julia": 0.0,
    "go": 0.0,
    "mojo": 1.0e-9,
}
SOURCES = (
    "benchmarks/bench_fitzhugh_nagumo_simulate.py",
    "engine/src/bindings/fitzhugh_nagumo.rs",
    "engine/src/neurons/simple_spiking/fitzhugh_nagumo.rs",
    "src/sc_neurocore/accel/go/neurons/fitzhugh_nagumo/fitzhugh_nagumo.go",
    "src/sc_neurocore/accel/go/services/fitzhugh_nagumo.go",
    "src/sc_neurocore/accel/go/services/fitzhugh_nagumo_test.go",
    "src/sc_neurocore/accel/julia/neurons/fitzhugh_nagumo.jl",
    "src/sc_neurocore/accel/mojo/neurons/fitzhugh_nagumo.mojo",
    "src/sc_neurocore/accel/rust/safety/fitzhugh_nagumo.rs",
    "src/sc_neurocore/neurons/model_schemas/fitzhugh_nagumo.json",
    "src/sc_neurocore/neurons/model_schemas/fitzhugh_nagumo.toml",
    "src/sc_neurocore/neurons/models/fitzhugh_nagumo.py",
)


class _RunRow(TypedDict):
    """One measured runtime result before JSON serialization."""

    available: bool
    used: bool
    median_ms: float
    min_ms: float
    results_ms: list[float]
    event_count: int
    final_state: list[float]
    trace: npt.NDArray[np.float64]


def _probe_rust() -> tuple[bool, str]:
    return (fhn._HAS_RUST, "" if fhn._HAS_RUST else "engine wheel lacks the symbol")


def _probe_julia() -> tuple[bool, str]:
    ok = fhn._ensure_julia_loaded()
    return (ok, "" if ok else "juliacall/.jl unavailable")


def _probe_go() -> tuple[bool, str]:
    ok = fhn._ensure_go_loaded()
    return (ok, "" if ok else "accel/go/neurons/fitzhugh_nagumo/libfhn.so not built")


def _probe_mojo() -> tuple[bool, str]:
    if not _os.path.isfile(_os.path.expanduser("~/.pixi/bin/mojo")):
        return False, "mojo binary not at ~/.pixi/bin/mojo"
    ok = fhn._ensure_mojo_loaded()
    return (ok, "" if ok else "accel/mojo/neurons/libfhn.so not built")


def _run(backend: str) -> _RunRow:
    FitzHughNagumoNeuron().simulate(N_STEPS, CURRENT, backend=backend)  # warm-up (Julia JIT)
    times_ms: list[float] = []
    traces: list[npt.NDArray[np.float64]] = []
    spike_counts: list[int] = []
    final_states: list[list[float]] = []
    for _ in range(N_REPEATS):
        neuron = FitzHughNagumoNeuron()
        t0 = time.perf_counter()
        trace, spikes = neuron.simulate(N_STEPS, CURRENT, backend=backend)
        times_ms.append((time.perf_counter() - t0) * 1000.0)
        traces.append(trace)
        spike_counts.append(spikes)
        final_states.append([neuron.v, neuron.w])
    if len(set(spike_counts)) != 1 or any(state != final_states[0] for state in final_states):
        raise RuntimeError(f"{backend} changed its result between benchmark repeats")
    ordered = sorted(times_ms)
    return {
        "available": True,
        "used": True,
        "median_ms": ordered[len(ordered) // 2],
        "min_ms": ordered[0],
        "results_ms": times_ms,
        "event_count": spike_counts[0],
        "final_state": final_states[0],
        "trace": traces[-1],
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="FitzHugh-Nagumo RK4 multi-language benchmark.")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    print("# FitzHugh-Nagumo RK4 N-step benchmark")
    print(f"# Workload: {N_STEPS:,} steps, default params, current={CURRENT}")
    print(f"# Repeats per backend: {N_REPEATS}")
    print(f"# Python: {platform.python_version()}, NumPy: {np.__version__}")
    print(f"# platform: {platform.platform()}")
    print("# isolation: non-isolated (loaded workstation) — functional/regression evidence")
    print()

    availability = {
        "python": (True, ""),
        "rust": _probe_rust(),
        "julia": _probe_julia(),
        "go": _probe_go(),
        "mojo": _probe_mojo(),
    }

    print(f"{'backend':<8}  {'available':<10}  reason")
    print(f"{'-' * 8}  {'-' * 10}  {'-' * 50}")
    for name, (avail, reason) in availability.items():
        print(f"{name:<8}  {'yes' if avail else 'no':<10}  {reason}")
    print()

    reference: npt.NDArray[np.float64] | None = None
    python_median: float | None = None
    reference_events: int | None = None
    reference_state: list[float] | None = None
    backends: dict[str, dict[str, object]] = {}

    print(f"{'backend':<8}  {'median ms':>12}  {'min ms':>12}  {'parity Δ':>12}  {'speedup':>9}")
    print(f"{'-' * 8}  {'-' * 12}  {'-' * 12}  {'-' * 12}  {'-' * 9}")
    for name, (avail, reason) in availability.items():
        if not avail:
            raise RuntimeError(f"required {name} backend is unavailable: {reason}")
        row = _run(name)
        trace = row["trace"]
        median_ms = row["median_ms"]
        min_ms = row["min_ms"]
        if name == "python":
            reference = trace
            python_median = median_ms
            parity = 0.0
            reference_events = row["event_count"]
            reference_state = list(row["final_state"])
        else:
            assert reference is not None and reference_events is not None
            assert reference_state is not None
            parity = float(np.max(np.abs(trace - reference)))
            state = row["final_state"]
            state_gap = max(
                abs(actual - expected) for actual, expected in zip(state, reference_state)
            )
            if parity > PARITY_ATOL[name] or row["event_count"] != reference_events:
                raise RuntimeError(f"{name} failed the enrolled FitzHugh-Nagumo parity contract")
            if state_gap > PARITY_ATOL[name]:
                raise RuntimeError(f"{name} failed the final-state parity contract")
        speedup = (python_median / median_ms) if python_median and median_ms > 0 else float("nan")
        print(f"{name:<8}  {median_ms:>12.2f}  {min_ms:>12.2f}  {parity:>12.2e}  {speedup:>8.2f}x")
        assert reference_events is not None and reference_state is not None
        final_state_matches = (
            max(
                abs(float(actual) - expected)
                for actual, expected in zip(row["final_state"], reference_state, strict=True)
            )
            <= PARITY_ATOL[name]
        )
        backends[name] = {
            "available": row["available"],
            "used": row["used"],
            "median_ms": row["median_ms"],
            "min_ms": row["min_ms"],
            "results_ms": row["results_ms"],
            "event_count": row["event_count"],
            "final_state": row["final_state"],
            "parity_max_abs_diff": parity,
            "speedup_vs_python": speedup,
            "event_count_matches_python": row["event_count"] == reference_events,
            "final_state_matches_python": final_state_matches,
        }

    print()
    print("# Note: the RHS is exact arithmetic (v*v*v, no transcendentals) and the")
    print("# 2-D flow is non-chaotic, so rust/julia/go reproduce the trace bit-for-bit")
    print("# (parity 0). Mojo fuses some RK4 multiply-adds into FMAs and sits within a")
    print("# small non-amplifying ULP band with identical spike counts. auto -> Rust.")

    report = {
        "schema_version": "sc-neurocore.polyglot-benchmark.v1",
        "model": "FitzHughNagumoNeuron",
        "benchmark": "fitzhugh_nagumo_simulate_rk4",
        "workload": {"n_steps": N_STEPS, "current": CURRENT, "repeats": N_REPEATS},
        "measured_order": list(BACKENDS),
        "backends": backends,
        "source_hashes": {
            source: hashlib.sha256((ROOT / source).read_bytes()).hexdigest() for source in SOURCES
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "single_cpu_pinned": len(_os.sched_getaffinity(0)) == 1,
            "isolation": "non-isolated (loaded workstation)",
        },
        "evidence_class": "local_regression_non_isolated",
        "production_speed_claim": False,
        "hardware_measurement_claimed": False,
        "notes": "Loaded-host regression only; timings are not comparative production claims.",
    }
    output = args.json if args.json is not None else OUTPUT
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nWrote {output}")
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))
