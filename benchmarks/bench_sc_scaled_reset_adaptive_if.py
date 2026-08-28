#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — retained scaled-reset adaptive-IF backend benchmark

"""Measure the count-neutral retained recurrence across all five runtimes."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from sc_neurocore.neurons.models import sc_scaled_reset_adaptive_if as implementation
from sc_neurocore.neurons.models.sc_scaled_reset_adaptive_if import (
    SCScaledResetAdaptiveIFNeuron,
)

N_STEPS = 200_000
CURRENT = 3.0
N_REPEATS = 3
MODEL_KWARGS = {
    "theta_reset": 1.3,
    "tau_theta": 40.0,
    "tau_1": 15.0,
    "tau_2": 80.0,
    "a": 0.1,
    "b": 0.1,
    "r1": 0.2,
    "r2": -0.15,
}


def _availability() -> dict[str, tuple[bool, str]]:
    probes = {
        "python": (True, ""),
        "rust": (implementation._HAS_RUST, "engine wheel lacks the symbol"),
        "julia": (
            implementation._ensure_julia_loaded(),
            "juliacall or Julia module unavailable",
        ),
        "go": (implementation._ensure_go_loaded(), "Go shared library unavailable"),
        "mojo": (implementation._ensure_mojo_loaded(), "Mojo shared library unavailable"),
    }
    return {
        name: (available, "" if available else reason)
        for name, (available, reason) in probes.items()
    }


def _run(backend: str) -> tuple[float, float, NDArray[np.float64], int]:
    SCScaledResetAdaptiveIFNeuron(**MODEL_KWARGS).simulate(N_STEPS, CURRENT, backend=backend)
    timings: list[float] = []
    trace: NDArray[np.float64] = np.empty((0, 4), dtype=np.float64)
    spikes = 0
    for _ in range(N_REPEATS):
        start = time.perf_counter()
        trace, spikes = SCScaledResetAdaptiveIFNeuron(**MODEL_KWARGS).simulate(
            N_STEPS, CURRENT, backend=backend
        )
        timings.append((time.perf_counter() - start) * 1000.0)
    timings.sort()
    return timings[len(timings) // 2], timings[0], trace, spikes


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    availability = _availability()
    reference: NDArray[np.float64] | None = None
    reference_spikes = 0
    python_median: float | None = None
    rows: list[dict[str, object]] = []
    for backend, (available, reason) in availability.items():
        if not available:
            rows.append({"backend": backend, "skipped": True, "unavailable_reason": reason})
            continue
        median_ms, minimum_ms, trace, spikes = _run(backend)
        if reference is None:
            reference = trace
            reference_spikes = spikes
            python_median = median_ms
        if reference is None or python_median is None:
            raise RuntimeError("Python reference must run before optional backends")
        rows.append(
            {
                "backend": backend,
                "median_ms": median_ms,
                "min_ms": minimum_ms,
                "parity_max_abs_diff": float(np.max(np.abs(trace - reference))),
                "event_delta": abs(spikes - reference_spikes),
                "speedup_vs_python": python_median / median_ms,
            }
        )
    report = {
        "benchmark": "sc_scaled_reset_adaptive_if",
        "workload": {"n_steps": N_STEPS, "current": CURRENT, "repeats": N_REPEATS},
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "isolation": "non-isolated (loaded workstation)",
        },
        "results": rows,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))
