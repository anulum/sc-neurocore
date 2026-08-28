#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — retained four-state GLIF five-runtime benchmark

"""Measure complete-state parity and loaded-host throughput for the SC identity."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import numpy.typing as npt

from sc_neurocore.neurons.models import sc_four_state_glif
from sc_neurocore.neurons.models.sc_four_state_glif import SCFourStateGLIFNeuron

N_STEPS = 2_000_000
CURRENT = 30.0
N_REPEATS = 3
REPOSITORY = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(
    backend: str,
) -> tuple[float, float, npt.NDArray[np.float64], int, dict[str, float]]:
    SCFourStateGLIFNeuron().simulate(1, CURRENT, backend=backend)
    times_ms: list[float] = []
    trace: npt.NDArray[np.float64] = np.empty(0, dtype=np.float64)
    for _ in range(N_REPEATS):
        neuron = SCFourStateGLIFNeuron()
        start = time.perf_counter()
        trace, events = neuron.simulate(N_STEPS, CURRENT, backend=backend)
        times_ms.append((time.perf_counter() - start) * 1000.0)
    times_ms.sort()
    state = {
        "v": neuron.v,
        "theta": neuron.theta,
        "i_asc1": neuron.i_asc1,
        "i_asc2": neuron.i_asc2,
    }
    return times_ms[len(times_ms) // 2], times_ms[0], trace, events, state


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json",
        type=Path,
        default=REPOSITORY / "benchmarks/results/bench_sc_four_state_glif.json",
    )
    args = parser.parse_args(argv)
    probes: dict[str, Callable[[], bool]] = {
        "python": lambda: True,
        "rust": lambda: sc_four_state_glif._HAS_RUST,
        "julia": sc_four_state_glif._ensure_julia_loaded,
        "go": sc_four_state_glif._ensure_go_loaded,
        "mojo": sc_four_state_glif._ensure_mojo_loaded,
    }
    rows: list[dict[str, object]] = []
    reference_trace: npt.NDArray[np.float64] | None = None
    reference_state: dict[str, float] | None = None
    reference_events: int | None = None
    python_median: float | None = None
    for backend, probe in probes.items():
        if not probe():
            rows.append({"backend": backend, "skipped": True})
            continue
        median_ms, min_ms, trace, events, state = _run(backend)
        if backend == "python":
            reference_trace = trace
            reference_state = state
            reference_events = events
            python_median = median_ms
            parity = 0.0
        else:
            assert reference_trace is not None
            assert reference_state is not None
            parity = max(
                float(np.max(np.abs(trace - reference_trace))),
                max(abs(state[key] - reference_state[key]) for key in state),
            )
        rows.append(
            {
                "backend": backend,
                "median_ms": median_ms,
                "min_ms": min_ms,
                "speedup_vs_python": python_median / median_ms if python_median else 1.0,
                "parity_max_abs_diff": parity,
                "events": events,
                "event_delta_vs_python": 0
                if reference_events is None
                else events - reference_events,
                "final_state": state,
            }
        )

    sources = {
        "python": "src/sc_neurocore/neurons/models/sc_four_state_glif.py",
        "rust": "engine/src/neurons/biophysical/sc_four_state_glif.rs",
        "julia": "src/sc_neurocore/accel/julia/neurons/sc_four_state_glif.jl",
        "go": "src/sc_neurocore/accel/go/neurons/sc_four_state_glif/sc_four_state_glif.go",
        "mojo": "src/sc_neurocore/accel/mojo/neurons/sc_four_state_glif.mojo",
        "receipt": "src/sc_neurocore/neurons/reference_receipts/sc_four_state_glif_project.json",
    }
    report = {
        "benchmark": "sc_four_state_glif",
        "workload": {"n_steps": N_STEPS, "current": CURRENT, "repeats": N_REPEATS},
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "isolation": "non-isolated loaded workstation",
            "hardware_measurement_claimed": False,
        },
        "source_sha256": {name: _sha256(REPOSITORY / path) for name, path in sources.items()},
        "results": rows,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for row in rows:
        print(row)
    print(f"Wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
