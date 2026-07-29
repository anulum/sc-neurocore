# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — source/binary-bound five-runtime Compte benchmark

"""Measure complete Compte trajectories without production-speed claims."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import tempfile
import time
from typing import cast

import numpy as np
import numpy.typing as npt

from sc_neurocore.accel import compte_wm as backends
from sc_neurocore.neurons.models.compte_wm import CompteWMNeuron

REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPOSITORY / "benchmarks/results/bench_compte_wm.json"
BACKENDS = ("python", "rust", "julia", "go", "mojo")
KERNEL = backends.KERNEL
PARITY_ATOL = backends.PARITY_ATOL
N_STEPS = 200_000
N_REPEATS = 3
SOURCE_PATHS = (
    "benchmarks/bench_compte_wm.py",
    "engine/src/bindings/rate/compte_wm.rs",
    "engine/src/neurons/rate/compte_wm.rs",
    "hdl/formal/catalogue/sc_compte_wm.v",
    "src/sc_neurocore/accel/compte_wm.py",
    "src/sc_neurocore/accel/go/compte_wm/compte_wm.go",
    "src/sc_neurocore/accel/go/services/compte_wm.go",
    "src/sc_neurocore/accel/julia/neurons/compte_wm.jl",
    "src/sc_neurocore/accel/mojo/compte_wm/compte_wm.mojo",
    "src/sc_neurocore/accel/rust/safety/compte_wm.rs",
    "src/sc_neurocore/neurons/model_descriptors/CompteWMNeuron.toml",
    "src/sc_neurocore/neurons/model_schemas/compte_wm.json",
    "src/sc_neurocore/neurons/model_schemas/compte_wm.toml",
    "src/sc_neurocore/neurons/models/compte_wm.py",
    "src/sc_neurocore/neurons/reference_trace_data/compte_wm_2000_pyramidal.json",
)
_TRACE_KEYS = ("voltages", "s_ampa", "s_nmda", "x_nmda", "s_gaba", "refractory")
FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_hashes() -> dict[str, str]:
    return {relative: _sha256(REPOSITORY / relative) for relative in SOURCE_PATHS}


def _binary_record(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    try:
        display = str(resolved.relative_to(REPOSITORY.resolve()))
    except ValueError:
        display = str(resolved)
    return {"path": display, "sha256": _sha256(resolved), "size_bytes": resolved.stat().st_size}


def _binary_hashes() -> dict[str, dict[str, object]]:
    extension = importlib.import_module("sc_neurocore_engine.sc_neurocore_engine")
    return {
        "rust_extension": _binary_record(Path(str(extension.__file__))),
        "go_shared_library": _binary_record(
            REPOSITORY / "src/sc_neurocore/accel/go/compte_wm/libcompte_wm.so"
        ),
        "mojo_shared_library": _binary_record(
            REPOSITORY / "src/sc_neurocore/accel/mojo/compte_wm/libcompte_wm.so"
        ),
    }


def _inputs(steps: int) -> tuple[FloatArray, IntArray, IntArray, IntArray]:
    index = np.arange(steps)
    return (
        1.0 + 0.2 * np.sin(index * 0.03),
        (index % 17 == 0).astype(np.int64),
        (index % 11 == 0).astype(np.int64),
        (index % 23 == 0).astype(np.int64),
    )


def _run_backend(backend: str, steps: int) -> backends.CompteWMResult:
    return cast(
        backends.CompteWMResult, CompteWMNeuron().simulate(*_inputs(steps), backend=backend)
    )


def _trace_hash(result: backends.CompteWMResult) -> str:
    values = np.column_stack(
        (
            *(cast(FloatArray, result[key]) for key in _TRACE_KEYS),
            cast(IntArray, result["events"]),
        )
    )
    return hashlib.sha256(np.ascontiguousarray(values, dtype="<f8").tobytes()).hexdigest()


def _measure(backend: str, steps: int, repeats: int) -> tuple[list[int], backends.CompteWMResult]:
    _run_backend(backend, min(256, steps))
    samples: list[int] = []
    result = _run_backend(backend, 0)
    for _ in range(repeats):
        start = time.perf_counter_ns()
        result = _run_backend(backend, steps)
        samples.append(time.perf_counter_ns() - start)
    return samples, result


def _tool_version(command: list[str]) -> str:
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=30)
    except OSError:
        return "unavailable"
    lines = (completed.stdout or completed.stderr).strip().splitlines()
    return lines[0] if lines else f"exit {completed.returncode}"


def _environment() -> dict[str, object]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else [],
        "load_average": list(os.getloadavg()) if hasattr(os, "getloadavg") else [],
        "rustc": _tool_version(["rustc", "--version"]),
        "go": _tool_version(["go", "version"]),
        "julia": _tool_version(["julia", "--version"]),
        "mojo": _tool_version(["mojo", "--version"]),
    }


def _verify_rust_safety() -> dict[str, object]:
    source = REPOSITORY / "src/sc_neurocore/accel/rust/safety/compte_wm.rs"
    with tempfile.TemporaryDirectory(prefix="compte-safety-") as directory:
        binary = Path(directory) / "tests"
        compiled = subprocess.run(
            ["rustc", "--edition", "2021", "--test", str(source), "-o", str(binary)],
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if compiled.returncode != 0:
            return {
                "passed": False,
                "returncode": compiled.returncode,
                "output_tail": compiled.stderr.splitlines()[-20:],
            }
        executed = subprocess.run(
            [str(binary)], check=False, capture_output=True, text=True, timeout=60
        )
    return {
        "passed": executed.returncode == 0,
        "returncode": executed.returncode,
        "output_tail": (executed.stdout + executed.stderr).splitlines()[-20:],
    }


def build_payload(steps: int, repeats: int) -> tuple[dict[str, object], bool]:
    """Measure all five real lanes and return their aggregate parity verdict."""
    reference: backends.CompteWMResult | None = None
    rows: dict[str, dict[str, object]] = {}
    order: list[tuple[int, str]] = []
    passed = True
    for backend in BACKENDS:
        if not backends.backend_available(backend):
            rows[backend] = {"available": False, "used": False, "reason": "runtime unavailable"}
            passed = False
            continue
        samples, result = _measure(backend, steps, repeats)
        if reference is None:
            reference = result
        assert reference is not None
        tolerance = PARITY_ATOL[backend]
        max_gap = 0.0
        for key in _TRACE_KEYS:
            max_gap = max(
                max_gap,
                float(
                    np.max(
                        np.abs(cast(FloatArray, result[key]) - cast(FloatArray, reference[key])),
                        initial=0.0,
                    )
                ),
            )
        event_exact = bool(np.array_equal(result["events"], reference["events"]))
        parity = max_gap <= tolerance and event_exact
        median = int(statistics.median(samples))
        order.append((median, backend))
        rows[backend] = {
            "available": True,
            "used": True,
            "samples_ns": samples,
            "median_ns": median,
            "ns_per_step": median / max(1, steps),
            "parity_max_abs_diff": max_gap,
            "parity_tolerance": tolerance,
            "events_exact": event_exact,
            "event_count": int(np.sum(cast(IntArray, result["events"]))),
            "trace_matches_python": parity,
            "trace_sha256": _trace_hash(result),
        }
        passed = passed and parity
    safety = _verify_rust_safety()
    passed = passed and bool(safety["passed"])
    payload: dict[str, object] = {
        "schema_version": "sc-neurocore.polyglot-benchmark.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "kernel": KERNEL,
        "model": "CompteWMNeuron",
        "evidence_class": "local_regression",
        "production_speed_claim": False,
        "hardware_measurement_claimed": False,
        "network_behavior_claimed": False,
        "configuration": {
            "steps": steps,
            "repeats": repeats,
            "dt_ms": 0.02,
            "event_periods": {"recurrent_nmda": 17, "external_ampa": 11, "inhibitory_gabaa": 23},
        },
        "meta": {
            "single_cpu_pinned": len(os.sched_getaffinity(0)) == 1,
            "exclusive_cpu_isolation_claimed": False,
        },
        "environment": _environment(),
        "source_hashes": _source_hashes(),
        "binary_hashes": _binary_hashes(),
        "rust_safety": safety,
        "results": rows,
        "observed_order_fastest_first": [backend for _, backend in sorted(order)],
        "passed": passed,
    }
    return payload, passed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=N_STEPS)
    parser.add_argument("--repeats", type=int, default=N_REPEATS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload, passed = build_payload(args.steps, args.repeats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
