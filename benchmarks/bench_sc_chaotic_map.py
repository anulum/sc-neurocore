# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — controlled SC chaotic-map polyglot benchmark

"""Measure all maintained lanes of the project SC chaotic map."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import statistics
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from sc_neurocore.accel import sc_chaotic_map

ROOT = Path(__file__).resolve().parents[1]
BACKENDS = ("python", "rust", "julia", "go", "mojo")
N_STEPS = 200_000
N_REPEATS = 3
SOURCE_PATHS = (
    "benchmarks/bench_sc_chaotic_map.py",
    "engine/src/neurons/sc_chaotic_map.rs",
    "src/sc_neurocore/accel/sc_chaotic_map.py",
    "src/sc_neurocore/accel/go/sc_chaotic_map/sc_chaotic_map.go",
    "src/sc_neurocore/accel/go/services/sc_chaotic_map_neuron.go",
    "src/sc_neurocore/accel/julia/neurons/sc_chaotic_map_neuron.jl",
    "src/sc_neurocore/accel/mojo/sc_chaotic_map/sc_chaotic_map.mojo",
    "src/sc_neurocore/accel/rust/safety/sc_chaotic_map_neuron.rs",
    "src/sc_neurocore/neurons/models/sc_chaotic_map_neuron.py",
)
BINARY_PATHS = (
    "src/sc_neurocore/accel/go/sc_chaotic_map/libsc_chaotic_map.so",
    "src/sc_neurocore/accel/mojo/sc_chaotic_map/libsc_chaotic_map.so",
)


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


def _hashes(paths: tuple[str, ...]) -> dict[str, str]:
    """Bind the report to each measured source or native library."""
    return {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in paths}


def _binary_hashes() -> dict[str, str]:
    """Bind the report to the loaded Rust, Go, and Mojo libraries."""
    paths = [ROOT / path for path in BINARY_PATHS]
    rust = tuple((ROOT / "bridge/sc_neurocore_engine").glob("sc_neurocore_engine*.so"))
    if len(rust) != 1:
        raise RuntimeError(f"expected one Rust extension, found {len(rust)}")
    paths.append(rust[0])
    return {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
    }


def _measure(
    backend: str, drive: npt.NDArray[np.float64]
) -> tuple[list[float], Mapping[str, object]]:
    """Warm one lane, then return controlled wall-clock samples and its receipt."""
    sc_chaotic_map.simulate_sc_chaotic_map(0.4, -0.2, current=drive, backend=backend)
    samples: list[float] = []
    receipt: Mapping[str, object] = {}
    for _ in range(N_REPEATS):
        gc.collect()
        started = time.perf_counter_ns()
        receipt = sc_chaotic_map.simulate_sc_chaotic_map(0.4, -0.2, current=drive, backend=backend)
        samples.append((time.perf_counter_ns() - started) / 1_000_000.0)
    return samples, receipt


def _trace_hash(receipt: Mapping[str, object]) -> str:
    """Hash complete binary64 states and byte events in a stable field order."""
    digest = hashlib.sha256()
    for field in ("x", "y"):
        digest.update(np.asarray(receipt[field], dtype="<f8").tobytes())
    digest.update(np.asarray(receipt["spikes"], dtype=np.uint8).tobytes())
    return digest.hexdigest()


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
    missing = [name for name in BACKENDS if not sc_chaotic_map.backend_available(name)]
    if missing:
        print(f"Missing required backends: {missing}")
        return 2

    index = np.arange(N_STEPS, dtype=np.int64)
    drive = np.where(index % 2 == 0, 1.0, -1.0)
    reference = sc_chaotic_map.simulate_sc_chaotic_map(0.4, -0.2, current=drive, backend="python")
    rows: dict[str, dict[str, Any]] = {}
    python_median = 0.0
    for backend in BACKENDS:
        samples, receipt = _measure(backend, drive)
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
            for field in ("x", "y")
        }
        events_exact = np.array_equal(
            np.asarray(receipt["spikes"]), np.asarray(reference["spikes"])
        )
        if max(deltas.values()) > sc_chaotic_map.PARITY_ATOL[backend] or not events_exact:
            raise RuntimeError(f"{backend} failed complete-trace parity")
        rows[backend] = {
            "samples_ms": samples,
            "median_call_ms": median,
            "minimum_call_ms": min(samples),
            "speedup_vs_python": python_median / median,
            "parity_max_abs_diff": deltas,
            "events_exact": events_exact,
            "event_count": int(cast(int, receipt["spike_count"])),
            "trajectory_sha256": _trace_hash(receipt),
        }

    report = {
        "schema_version": "sc-neurocore.sc-chaotic-map-polyglot-benchmark.v1",
        "identity": "SC-NeuroCore project model; no publication attribution",
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
        "backends": rows,
        "measured_order": sorted(BACKENDS, key=lambda name: rows[name]["median_call_ms"]),
        "source_hashes": _hashes(SOURCE_PATHS),
        "binary_hashes": _binary_hashes(),
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for backend in report["measured_order"]:
        row = rows[backend]
        print(f"{backend:>7}: {row['median_call_ms']:.3f} ms, events={row['event_count']}")
    print(f"Wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
