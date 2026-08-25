# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — controlled Aihara polyglot benchmark

"""Benchmark every Aihara lane at the source Figure 4 periodic point."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np

from sc_neurocore.accel import aihara_map

ROOT = Path(__file__).resolve().parents[1]
KERNEL = aihara_map.KERNEL
BACKENDS = ("python", "rust", "julia", "go", "mojo")
N_STEPS = 200_000
N_REPEATS = 3
BIAS = 0.6288
SOURCE_PATHS = (
    "benchmarks/bench_aihara_map.py",
    "engine/src/neurons/aihara_map.rs",
    "engine/src/bindings/maps/aihara_map.rs",
    "src/sc_neurocore/accel/aihara_map.py",
    "src/sc_neurocore/accel/go/aihara_map/aihara_map.go",
    "src/sc_neurocore/accel/julia/neurons/aihara_map_neuron.jl",
    "src/sc_neurocore/accel/mojo/aihara_map/aihara_map.mojo",
    "src/sc_neurocore/neurons/models/aihara_map_neuron.py",
)


def _hashes() -> dict[str, str]:
    return {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in SOURCE_PATHS}


def _cpu_model() -> str:
    try:
        lines = Path("/proc/cpuinfo").read_text().splitlines()
    except OSError:
        return platform.processor() or "unknown"
    for line in lines:
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def _measure(backend: str, drive: np.ndarray) -> tuple[list[float], dict[str, object]]:
    aihara_map.simulate_aihara_map(bias=BIAS, current=drive[:2000], backend=backend)
    samples: list[float] = []
    result: dict[str, object] = {}
    for _ in range(N_REPEATS):
        gc.collect()
        started = time.perf_counter_ns()
        result = dict(aihara_map.simulate_aihara_map(bias=BIAS, current=drive, backend=backend))
        samples.append((time.perf_counter_ns() - started) / 1_000_000.0)
    return samples, result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--allow-unpinned", action="store_true")
    args = parser.parse_args(argv)
    affinity = sorted(os.sched_getaffinity(0))
    if len(affinity) != 1 and not args.allow_unpinned:
        print(f"Refusing unpinned benchmark; affinity is {affinity}")
        return 2
    unavailable = [backend for backend in BACKENDS if not aihara_map.backend_available(backend)]
    if unavailable:
        print("Missing required backend(s): " + ", ".join(unavailable))
        return 2

    drive = np.zeros(N_STEPS, dtype=np.float64)
    rows: dict[str, dict[str, Any]] = {}
    reference: dict[str, object] | None = None
    python_median = 0.0
    for backend in BACKENDS:
        samples, result = _measure(backend, drive)
        median = statistics.median(samples)
        if backend == "python":
            reference = result
            python_median = median
        assert reference is not None
        y_delta = float(
            np.max(
                np.abs(
                    np.asarray(result["y"], dtype=np.float64)
                    - np.asarray(reference["y"], dtype=np.float64)
                )
            )
        )
        events_match = int(result["spike_count"]) == int(reference["spike_count"])
        rows[backend] = {
            "available": True,
            "samples_ms": samples,
            "median_call_ms": median,
            "minimum_call_ms": min(samples),
            "speedup_vs_python": python_median / median,
            "parity_max_abs_diff": y_delta,
            "event_count": int(result["spike_count"]),
            "event_count_matches_python": events_match,
            "y_final": float(result["y_final"]),
        }
        if y_delta > aihara_map.PARITY_ATOL[backend] or not events_match:
            print(f"{backend} failed Figure 4 periodic parity")
            return 3

    measured_order = sorted(BACKENDS, key=lambda name: rows[name]["median_call_ms"])
    report = {
        "schema_version": "sc-neurocore.polyglot-benchmark.v1",
        "kernel": KERNEL,
        "workload": {
            "n_steps": N_STEPS,
            "repeats": N_REPEATS,
            "current": 0.0,
            "bias": BIAS,
            "source_anchor": "Aihara 1989 Figure 4 periodic operating point",
        },
        "environment": {
            "cpu": _cpu_model(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "affinity": affinity,
            "single_cpu_pinned": len(affinity) == 1,
            "load_average": list(os.getloadavg()),
            "scope": "local diagnostic regression evidence; not a portable ranking",
        },
        "backends": rows,
        "measured_order": measured_order,
        "source_hashes": _hashes(),
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    for backend in measured_order:
        row = rows[backend]
        print(
            f"{backend:>7}: {row['median_call_ms']:.3f} ms "
            f"{row['speedup_vs_python']:.2f}x events={row['event_count']}"
        )
    print(f"Wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
