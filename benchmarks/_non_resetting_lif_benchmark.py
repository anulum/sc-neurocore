# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""Shared source/binary-bound benchmark runner for the Model 50 identity pair."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time
from typing import Any, TypeAlias, cast

import numpy as np
import numpy.typing as npt

REPOSITORY = Path(__file__).resolve().parents[1]
BACKENDS = ("python", "rust", "julia", "go", "mojo")
FloatArray: TypeAlias = npt.NDArray[np.float64]


@dataclass(frozen=True)
class BenchmarkSpec:
    """Complete evidence-writing contract for one explicit model identity."""

    benchmark: str
    model: str
    output: Path
    current: float
    steps: int
    repeats: int
    simulate: Callable[..., Mapping[str, object]]
    backend_available: Callable[[str], bool]
    parity_atol: Mapping[str, float]
    state_keys: tuple[str, ...]
    final_keys: tuple[str, ...]
    source_paths: tuple[str, ...]
    go_library: str
    mojo_library: str


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_hashes(paths: Sequence[str]) -> dict[str, object]:
    hashes: dict[str, object] = {}
    for relative in paths:
        digest = _sha256(REPOSITORY / relative)
        hashes[relative] = digest
        stem, suffix = relative.rsplit(".", 1)
        aliases = hashes.setdefault(stem, {})
        if not isinstance(aliases, dict):
            raise RuntimeError(f"source-hash alias collision at {stem}")
        aliases[suffix] = digest
    return hashes


def _binary_hashes(spec: BenchmarkSpec) -> dict[str, dict[str, Any]]:
    extension = importlib.import_module("sc_neurocore_engine.sc_neurocore_engine")
    paths = {
        "rust_extension": Path(str(extension.__file__)),
        "go_shared_library": REPOSITORY / spec.go_library,
        "mojo_shared_library": REPOSITORY / spec.mojo_library,
    }
    return {
        name: {
            "path": (
                str(path.resolve().relative_to(REPOSITORY.resolve()))
                if path.resolve().is_relative_to(REPOSITORY.resolve())
                else f"$WHEEL_SITE/sc_neurocore_engine/{path.name}"
            ),
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for name, path in paths.items()
    }


def _trace_hash(result: Mapping[str, object], state_keys: Sequence[str]) -> str:
    states = np.ascontiguousarray(
        np.column_stack([np.asarray(result[key], dtype="<f8") for key in state_keys]),
        dtype="<f8",
    )
    events = np.ascontiguousarray(result["events"], dtype=np.uint8)
    digest = hashlib.sha256(states.tobytes())
    digest.update(events.tobytes())
    return digest.hexdigest()


def _measure(
    spec: BenchmarkSpec, backend: str, currents: FloatArray, repeats: int
) -> tuple[list[int], Mapping[str, object]]:
    spec.simulate(currents[: min(1000, len(currents))], backend=backend)
    samples: list[int] = []
    result: Mapping[str, object] = {}
    for _ in range(repeats):
        started = time.perf_counter_ns()
        result = spec.simulate(currents, backend=backend)
        samples.append(time.perf_counter_ns() - started)
    return samples, result


def _row(
    spec: BenchmarkSpec,
    backend: str,
    samples: list[int],
    result: Mapping[str, object],
    reference: Mapping[str, object],
    steps: int,
) -> dict[str, Any]:
    differences = [
        np.abs(np.asarray(result[key]) - np.asarray(reference[key])) for key in spec.state_keys
    ]
    maximum = max(float(values.max(initial=0.0)) for values in differences)
    event_match = np.array_equal(
        np.asarray(result["events"]), np.asarray(reference["events"])
    )
    median_ns = float(statistics.median(samples))
    event_count = int(np.asarray(result["events"]).sum())
    return {
        "available": True,
        "samples_ns": samples,
        "median_ns_per_step": median_ns / steps,
        "min_ns_per_step": min(samples) / steps,
        "max_ns_per_step": max(samples) / steps,
        "trace_sha256": _trace_hash(result, spec.state_keys),
        "parity_atol": spec.parity_atol[backend],
        "parity_max_abs_diff": maximum,
        "trace_matches_python": maximum <= spec.parity_atol[backend] and event_match,
        "events": event_count,
        "spikes": event_count,
        "event_vector_matches_python": event_match,
        "final_state": [float(cast(float, result[key])) for key in spec.final_keys],
    }


def _version(command: list[str]) -> str:
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=30, check=False)
    except OSError:
        return "unavailable"
    lines = (completed.stdout or completed.stderr).strip().splitlines()
    return lines[0] if lines else f"exit {completed.returncode}"


def run(spec: BenchmarkSpec, argv: list[str] | None = None) -> int:
    """Execute all five real runtimes and write evidence only after parity."""
    parser = argparse.ArgumentParser(description=spec.benchmark)
    parser.add_argument("--steps", type=int, default=spec.steps)
    parser.add_argument("--repeats", type=int, default=spec.repeats)
    parser.add_argument("--output", type=Path, default=spec.output)
    args = parser.parse_args(argv)
    if args.steps <= 0 or args.repeats <= 0:
        parser.error("steps and repeats must be positive")
    unavailable = [backend for backend in BACKENDS if not spec.backend_available(backend)]
    if unavailable:
        print(f"unavailable {spec.model} backends: {', '.join(unavailable)}")
        return 2
    currents = np.full(args.steps, spec.current, dtype=np.float64)
    measurements = {
        backend: _measure(spec, backend, currents, args.repeats) for backend in BACKENDS
    }
    reference = measurements["python"][1]
    rows = {
        backend: _row(spec, backend, samples, result, reference, args.steps)
        for backend, (samples, result) in measurements.items()
    }
    if not all(row["trace_matches_python"] for row in rows.values()):
        print(f"{spec.model} parity failed; evidence was not written")
        return 3
    payload = {
        "schema_version": "sc-neurocore.polyglot-benchmark.v1",
        "benchmark": spec.benchmark,
        "model": spec.model,
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
        "current": spec.current,
        "backend_summary": rows,
        "source_hashes": _source_hashes(spec.source_paths),
        "binary_hashes": _binary_hashes(spec),
        "tool_versions": {
            "rustc": _version([str(REPOSITORY / ".venv/bin/rustc"), "--version"]),
            "go": _version([str(REPOSITORY / ".venv/bin/go"), "version"]),
            "julia": _version([str(REPOSITORY / ".venv/bin/julia"), "--version"]),
            "mojo": _version([str(REPOSITORY / ".venv/bin/mojo"), "--version"]),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0
