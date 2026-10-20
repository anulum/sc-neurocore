#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — source-bound five-runtime Kobayashi MAT* benchmark

"""Measure all executable MAT* runtimes and publish only parity-clean evidence."""

from __future__ import annotations

from collections.abc import Mapping

import argparse
from datetime import datetime, timezone
import hashlib
from functools import partial
import importlib
import json
import os
from pathlib import Path
import platform
import re
import statistics
from typing import Any

import numpy as np
import numpy.typing as npt

from benchmarks._benchmark_context import load_average, measurement_context
from benchmarks._non_resetting_lif_benchmark import toolchain_versions
from benchmarks._repeated_measurements import (
    RepetitionSummary,
    compare_trajectories,
    measure_repetitions,
    summarize_repetitions,
)

from sc_neurocore.accel import mat as backends

REPOSITORY = Path(__file__).resolve().parents[1]
OUTPUT = REPOSITORY / "benchmarks/results/bench_mat.json"
BACKENDS = ("python", "rust", "julia", "go", "mojo")
STEPS = 200_000
REPEATS = 5
CURRENT = 0.7
GO_BENCH_RE = re.compile(r"^BenchmarkMATSource(?:-\d+)?\s+\d+\s+([0-9.]+)\s+ns/op")
SOURCE_PATHS = (
    "benchmarks/_repeated_measurements.py",
    "benchmarks/_non_resetting_lif_benchmark.py",
    "benchmarks/_benchmark_context.py",
    "benchmarks/bench_model_mat.py",
    "bridge/sc_neurocore_engine/__init__.py",
    "engine/src/bindings/trivial/mat.rs",
    "engine/src/neurons/trivial/mat.rs",
    "src/sc_neurocore/accel/mat.py",
    "src/sc_neurocore/accel/go/mat/mat.go",
    "src/sc_neurocore/accel/go/services/mat.go",
    "src/sc_neurocore/accel/go/services/mat_test.go",
    "src/sc_neurocore/accel/julia/neurons/mat.jl",
    "src/sc_neurocore/accel/mojo/kernels/mat.mojo",
    "src/sc_neurocore/accel/mojo/mat/mat.mojo",
    "src/sc_neurocore/accel/rust/safety/mat.rs",
    "src/sc_neurocore/neurons/model_descriptors/MATNeuron.toml",
    "src/sc_neurocore/neurons/model_schemas/mat.json",
    "src/sc_neurocore/neurons/model_schemas/mat.toml",
    "src/sc_neurocore/neurons/models/mat.py",
    "src/sc_neurocore/neurons/reference_trace_data/mat_2009_rs.json",
)

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
MATResult = dict[str, FloatArray | IntArray | float]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_hashes() -> dict[str, object]:
    hashes: dict[str, object] = {}
    for relative in SOURCE_PATHS:
        digest = _sha256(REPOSITORY / relative)
        hashes[relative] = digest
        stem, suffix = relative.rsplit(".", 1)
        aliases = hashes.setdefault(stem, {})
        if not isinstance(aliases, dict):
            raise RuntimeError(f"source-hash alias collision at {stem}")
        aliases[suffix] = digest
    return hashes


def _binary_hashes() -> dict[str, dict[str, Any]]:
    extension = importlib.import_module("sc_neurocore_engine.sc_neurocore_engine")
    paths = {
        "rust_extension": Path(str(extension.__file__)),
        "go_shared_library": REPOSITORY / "src/sc_neurocore/accel/go/mat/libmat.so",
        "mojo_shared_library": REPOSITORY / "src/sc_neurocore/accel/mojo/mat/libmat.so",
    }
    return {
        name: {
            "path": str(path.resolve().relative_to(REPOSITORY.resolve()))
            if path.resolve().is_relative_to(REPOSITORY.resolve())
            else f"$WHEEL_SITE/sc_neurocore_engine/{path.name}",
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for name, path in paths.items()
    }


def _trace_hash(result: Mapping[str, object]) -> str:
    states = np.ascontiguousarray(
        np.column_stack(
            [
                np.asarray(result[key], dtype="<f8")
                for key in ("voltages", "theta1", "theta2", "refractory")
            ]
        ),
        dtype="<f8",
    )
    events = np.ascontiguousarray(result["events"], dtype=np.uint8)
    digest = hashlib.sha256(states.tobytes())
    digest.update(events.tobytes())
    return digest.hexdigest()


def _row(
    backend: str,
    samples: list[int],
    result: Mapping[str, object],
    reference: Mapping[str, object],
    steps: int,
) -> RepetitionSummary:
    tolerance = backends.PARITY_ATOL[backend]
    max_difference, event_match, trace_match = compare_trajectories(
        result, reference, ("voltages", "theta1", "theta2", "refractory"), tolerance
    )
    median_ns = float(statistics.median(samples))
    return {
        "available": True,
        "samples_ns": samples,
        "median_ns_per_step": median_ns / steps,
        "min_ns_per_step": min(samples) / steps,
        "max_ns_per_step": max(samples) / steps,
        "trace_sha256": _trace_hash(result) if trace_match else None,
        "parity_atol": tolerance,
        "parity_max_abs_diff": max_difference,
        "trace_matches_python": trace_match,
        "events": int(np.asarray(result["events"]).sum()),
        "spikes": int(np.asarray(result["events"]).sum()),
        "event_vector_matches_python": event_match,
        "final_state": [
            float(np.asarray(result[key])[-1])
            for key in ("voltages", "theta1", "theta2", "refractory")
        ],
    }


def main(argv: list[str] | None = None) -> int:
    """Write evidence only when every runtime repetition matches the reference.

    Parameters
    ----------
    argv : list of str or None
        Command-line arguments, or the process arguments when omitted.

    Returns
    -------
    int
        Zero after writing evidence, or nonzero for unavailable or unequal runs.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    if args.steps <= 0 or args.repeats <= 0:
        parser.error("steps and repeats must be positive")
    currents = np.full(args.steps, CURRENT, dtype=np.float64)
    unavailable = [name for name in BACKENDS if not backends.backend_available(name)]
    if unavailable:
        print(f"unavailable MAT* backends: {', '.join(unavailable)}")
        return 2
    load_before = load_average()
    measurements = {
        name: measure_repetitions(
            backends.simulate_mat, currents, backend=name, repeats=args.repeats
        )
        for name in BACKENDS
    }
    reference = measurements["python"][1][0]
    rows = {
        name: summarize_repetitions(
            samples, results, partial(_row, name, reference=reference, steps=args.steps)
        )
        for name, (samples, results) in measurements.items()
    }
    if not all(row["trace_matches_python"] for row in rows.values()):
        print("MAT* parity failed; evidence was not written")
        return 3
    payload = {
        "schema_version": "sc-neurocore.polyglot-benchmark.v1",
        "benchmark": "Kobayashi MAT* non-resetting source recurrence",
        "model": "MATNeuron",
        "evidence_class": "local_regression_non_isolated",
        "production_speed_claim": False,
        "hardware_measurement_claimed": False,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else [],
        "load_average": list(os.getloadavg()) if hasattr(os, "getloadavg") else [],
        "steps": args.steps,
        "repeats": args.repeats,
        "current": CURRENT,
        "backend_summary": rows,
        "measurement_context": measurement_context(load_before),
        "source_hashes": _source_hashes(),
        "binary_hashes": _binary_hashes(),
        "tool_versions": toolchain_versions(REPOSITORY / "src/sc_neurocore/accel/go/mat/libmat.so"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
