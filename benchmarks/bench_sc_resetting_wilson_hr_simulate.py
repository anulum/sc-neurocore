#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — retained resetting Wilson-HR multi-language benchmark

"""Measure and source-bind every runtime for the retained SC recurrence."""

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

from sc_neurocore.neurons.models import sc_resetting_wilson_hr as implementation
from sc_neurocore.neurons.models.sc_resetting_wilson_hr import SCResettingWilsonHRNeuron

N_STEPS = 2_000_000
CURRENT = 2.0
N_REPEATS = 5
ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    "benchmarks/bench_sc_resetting_wilson_hr_simulate.py",
    "engine/src/bindings/sc_resetting_wilson_hr.rs",
    "engine/src/neurons/simple_spiking/sc_resetting_wilson_hr.rs",
    "src/sc_neurocore/accel/go/neurons/sc_resetting_wilson_hr/sc_resetting_wilson_hr.go",
    "src/sc_neurocore/accel/julia/neurons/sc_resetting_wilson_hr.jl",
    "src/sc_neurocore/accel/mojo/neurons/sc_resetting_wilson_hr.mojo",
    "src/sc_neurocore/accel/rust/safety/sc_resetting_wilson_hr.rs",
    "src/sc_neurocore/neurons/model_descriptors/SCResettingWilsonHRNeuron.toml",
    "src/sc_neurocore/neurons/model_schemas/sc_resetting_wilson_hr.json",
    "src/sc_neurocore/neurons/model_schemas/sc_resetting_wilson_hr.toml",
    "src/sc_neurocore/neurons/models/sc_resetting_wilson_hr.py",
    "src/sc_neurocore/neurons/reference_trace_data/sc_resetting_wilson_hr_project.json",
)


def _source_hashes() -> dict[str, object]:
    """Return flat source digests and suffix aliases for the evidence gate."""
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
    available = implementation._HAS_RUST
    return available, "" if available else "engine wheel lacks the retained SC symbol"


def _probe_julia() -> tuple[bool, str]:
    available = implementation._ensure_julia_loaded()
    return available, "" if available else "juliacall or retained SC Julia module unavailable"


def _probe_go() -> tuple[bool, str]:
    available = implementation._ensure_go_loaded()
    reason = "" if available else "Go retained SC shared library not built"
    return available, reason


def _probe_mojo() -> tuple[bool, str]:
    available = implementation._ensure_mojo_loaded()
    reason = "" if available else "Mojo retained SC shared library not built"
    return available, reason


def _run(backend: str) -> tuple[float, float, npt.NDArray[np.float64], int]:
    SCResettingWilsonHRNeuron().simulate(N_STEPS, CURRENT, backend=backend)
    measurements: list[float] = []
    trace: npt.NDArray[np.float64] = np.empty(0, dtype=np.float64)
    events = 0
    for _ in range(N_REPEATS):
        started = time.perf_counter()
        trace, events = SCResettingWilsonHRNeuron().simulate(N_STEPS, CURRENT, backend=backend)
        measurements.append((time.perf_counter() - started) * 1000.0)
    measurements.sort()
    return measurements[len(measurements) // 2], measurements[0], trace, events


def main(argv: list[str]) -> int:
    """Run the benchmark and optionally write the source-bound JSON receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)
    backends = {
        "python": (True, ""),
        "rust": _probe_rust(),
        "julia": _probe_julia(),
        "go": _probe_go(),
        "mojo": _probe_mojo(),
    }
    unavailable = {name: reason for name, (available, reason) in backends.items() if not available}
    if unavailable:
        print(f"Required backend unavailable; evidence was not written: {unavailable}")
        return 2

    reference: npt.NDArray[np.float64] | None = None
    reference_events: int | None = None
    python_median: float | None = None
    rows: dict[str, dict[str, float | int]] = {}
    for name in backends:
        median_ms, minimum_ms, trace, events = _run(name)
        if name == "python":
            reference = trace
            reference_events = events
            python_median = median_ms
            parity = 0.0
        else:
            assert reference is not None
            parity = float(np.max(np.abs(trace - reference)))
            if events != reference_events:
                raise RuntimeError(
                    f"SC resetting Wilson-HR {name} events {events} != Python {reference_events}"
                )
        speedup = python_median / median_ms if python_median and median_ms > 0.0 else float("nan")
        rows[name] = {
            "median_ms": median_ms,
            "min_ms": minimum_ms,
            "parity_max_abs_diff": parity,
            "event_count": events,
            "speedup_vs_python": speedup,
        }
        print(
            f"{name}: median={median_ms:.2f} ms, min={minimum_ms:.2f} ms, "
            f"events={events}, max_abs_diff={parity:.3e}"
        )

    report = {
        "schema_version": "sc-neurocore.polyglot-benchmark.v1",
        "benchmark": "sc_resetting_wilson_hr_simulate_rk4",
        "model": "SCResettingWilsonHRNeuron",
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
        print(f"Wrote {args.json}")
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))
