# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — controlled two-map polyglot benchmark

"""Benchmark all maintained lanes for the Nagumo-Sato and retained SC maps."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import statistics
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import numpy.typing as npt

from sc_neurocore.accel import nagumo_sato_map, sc_adaptive_threshold_map

ROOT = Path(__file__).resolve().parents[1]
BACKENDS = ("python", "rust", "julia", "go", "mojo")
N_STEPS = 200_000
N_REPEATS = 3
SOURCE_PATHS = (
    "benchmarks/bench_nagumo_sato_and_sc_adaptive_map.py",
    "engine/src/neurons/nagumo_sato_map.rs",
    "engine/src/neurons/sc_adaptive_threshold_map.rs",
    "src/sc_neurocore/accel/nagumo_sato_map.py",
    "src/sc_neurocore/accel/sc_adaptive_threshold_map.py",
    "src/sc_neurocore/accel/go/nagumo_sato_map/nagumo_sato_map.go",
    "src/sc_neurocore/accel/go/sc_adaptive_threshold_map/sc_adaptive_threshold_map.go",
    "src/sc_neurocore/accel/julia/neurons/nagumo_sato_map_neuron.jl",
    "src/sc_neurocore/accel/julia/neurons/sc_adaptive_threshold_map_neuron.jl",
    "src/sc_neurocore/accel/mojo/nagumo_sato_map/nagumo_sato_map.mojo",
    "src/sc_neurocore/accel/mojo/sc_adaptive_threshold_map/sc_adaptive_threshold_map.mojo",
    "src/sc_neurocore/neurons/models/nagumo_sato_map_neuron.py",
    "src/sc_neurocore/neurons/models/sc_adaptive_threshold_map_neuron.py",
)
BINARY_PATHS = (
    "src/sc_neurocore/accel/go/nagumo_sato_map/libnagumo_sato_map.so",
    "src/sc_neurocore/accel/go/sc_adaptive_threshold_map/libsc_adaptive_threshold_map.so",
    "src/sc_neurocore/accel/mojo/nagumo_sato_map/libnagumo_sato_map.so",
    "src/sc_neurocore/accel/mojo/sc_adaptive_threshold_map/libsc_adaptive_threshold_map.so",
)


class _Simulator(Protocol):
    """Common keyword-only surface of the two checked batch dispatchers."""

    def __call__(self, *, current: npt.ArrayLike, backend: str) -> Mapping[str, object]: ...


def _cpu_model() -> str:
    """Return the first Linux CPU model string, or a portable fallback."""
    try:
        lines = Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines()
    except OSError:
        return platform.processor() or "unknown"
    for line in lines:
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def _hashes() -> dict[str, str]:
    """Bind the report to every source that implements the measured maps."""
    return {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in SOURCE_PATHS}


def _measure(
    run: Callable[[], Mapping[str, object]],
) -> tuple[list[float], Mapping[str, object]]:
    """Warm one lane, then return controlled wall-clock samples and its receipt."""
    run()
    samples: list[float] = []
    receipt: Mapping[str, object] = {}
    for _ in range(N_REPEATS):
        gc.collect()
        started = time.perf_counter_ns()
        receipt = run()
        samples.append((time.perf_counter_ns() - started) / 1_000_000.0)
    return samples, receipt


def _benchmark_model(
    *,
    drive: npt.NDArray[np.float64],
    simulate: _Simulator,
    state_fields: tuple[str, ...],
    tolerance: dict[str, float],
) -> dict[str, Any]:
    """Measure one model and enforce complete state and event parity."""
    reference = simulate(current=drive, backend="python")
    rows: dict[str, dict[str, Any]] = {}
    python_median = 0.0
    for backend in BACKENDS:

        def run_lane() -> Mapping[str, object]:
            return simulate(current=drive, backend=backend)

        samples, receipt = _measure(run_lane)
        median = statistics.median(samples)
        if backend == "python":
            python_median = median
        deltas = {
            field: float(
                np.max(
                    np.abs(
                        np.asarray(receipt[field], dtype=np.float64)
                        - np.asarray(reference[field], dtype=np.float64)
                    )
                )
            )
            for field in state_fields
        }
        events_exact = np.array_equal(
            np.asarray(receipt["spikes"]), np.asarray(reference["spikes"])
        )
        if max(deltas.values(), default=0.0) > tolerance[backend] or not events_exact:
            raise RuntimeError(f"{backend} failed controlled complete-trace parity")
        rows[backend] = {
            "samples_ms": samples,
            "median_call_ms": median,
            "minimum_call_ms": min(samples),
            "speedup_vs_python": python_median / median,
            "parity_max_abs_diff": deltas,
            "events_exact": events_exact,
            "event_count": int(cast(int, receipt["spike_count"])),
        }
    return {
        "backends": rows,
        "measured_order": sorted(BACKENDS, key=lambda name: rows[name]["median_call_ms"]),
    }


def _binary_hashes() -> dict[str, str]:
    """Bind the evidence to the loaded Rust and built Go/Mojo libraries."""
    paths = [ROOT / path for path in BINARY_PATHS]
    rust_candidates = tuple((ROOT / "bridge/sc_neurocore_engine").glob("sc_neurocore_engine*.so"))
    if len(rust_candidates) != 1:
        raise RuntimeError(f"expected one Rust extension, found {len(rust_candidates)}")
    paths.append(rust_candidates[0])
    return {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
    }


def main(argv: list[str] | None = None) -> int:
    """Write a source-bound single-CPU benchmark report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--allow-unpinned", action="store_true")
    args = parser.parse_args(argv)
    affinity = sorted(os.sched_getaffinity(0))
    if len(affinity) != 1 and not args.allow_unpinned:
        print(f"Refusing unpinned benchmark; affinity is {affinity}")
        return 2
    missing = {
        model: [backend for backend in BACKENDS if not available(backend)]
        for model, available in (
            ("nagumo_sato_map", nagumo_sato_map.backend_available),
            ("sc_adaptive_threshold_map", sc_adaptive_threshold_map.backend_available),
        )
    }
    if any(missing.values()):
        print(f"Missing required backends: {missing}")
        return 2

    index = np.arange(N_STEPS, dtype=np.float64)
    models = {
        "nagumo_sato_map": _benchmark_model(
            drive=0.05 * np.sin(index * 0.037),
            simulate=nagumo_sato_map.simulate_nagumo_sato_map,
            state_fields=("y", "x"),
            tolerance=nagumo_sato_map.PARITY_ATOL,
        ),
        "sc_adaptive_threshold_map": _benchmark_model(
            drive=0.6 + 0.25 * np.sin(index * 0.017),
            simulate=sc_adaptive_threshold_map.simulate_sc_adaptive_threshold_map,
            state_fields=("x", "theta"),
            tolerance=sc_adaptive_threshold_map.PARITY_ATOL,
        ),
    }
    report = {
        "schema_version": "sc-neurocore.two-map-polyglot-benchmark.v1",
        "workload": {"n_steps": N_STEPS, "repeats": N_REPEATS},
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
        "models": models,
        "source_hashes": _hashes(),
        "binary_hashes": _binary_hashes(),
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for model, result in models.items():
        print(model)
        for backend in result["measured_order"]:
            row = result["backends"][backend]
            print(f"  {backend:>7}: {row['median_call_ms']:.3f} ms, events={row['event_count']}")
    print(f"Wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
