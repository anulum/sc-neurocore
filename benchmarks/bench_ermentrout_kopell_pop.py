# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source-bound five-runtime MPR benchmark

"""Measure and fail-closed verify the complete Montbrió–Pazó–Roxin runtime set."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import tempfile
import time
from typing import Any, SupportsFloat, TypeAlias, cast

import numpy as np
import numpy.typing as npt

from sc_neurocore.accel import ermentrout_kopell_pop as backends
from sc_neurocore.accel.backend_selection import current_cpu
from sc_neurocore.neurons.models.ermentrout_kopell_pop import (
    ErmentroutKopellPopulation,
)

REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPOSITORY / "benchmarks/results/bench_ermentrout_kopell_pop.json"
KERNEL = backends.KERNEL
BACKENDS = ("python", "rust", "julia", "go", "mojo")
N_STEPS = 50_000
N_REPEATS = 5
WARMUP_STEPS = 1_000
PARITY_ATOL = backends.PARITY_ATOL
TRACE_KEYS = ("r", "v")
FINAL_KEYS = tuple(f"{key}_final" for key in TRACE_KEYS)
INITIAL_STATE = {
    "r": 0.13,
    "v": -1.7,
    "tau": 1.3,
    "delta": 0.8,
    "eta_bar": -4.2,
    "j": 12.5,
    "dt": 0.004,
}
SOURCE_PATHS = (
    "Cargo.lock",
    "benchmarks/bench_ermentrout_kopell_pop.py",
    "bridge/sc_neurocore_engine/__init__.py",
    "engine/Cargo.toml",
    "engine/src/bindings/ermentrout_kopell_pop.rs",
    "engine/src/neurons/ermentrout_kopell_pop.rs",
    "engine/src/neurons/mod.rs",
    "engine/src/neurons/special.rs",
    "engine/src/pyo3_neurons.rs",
    "pyproject.toml",
    "src/sc_neurocore/accel/go/go.mod",
    "src/sc_neurocore/accel/go/ermentrout_kopell_pop/__init__.py",
    "src/sc_neurocore/accel/go/ermentrout_kopell_pop/ermentrout_kopell_pop.go",
    "src/sc_neurocore/accel/go/ermentrout_kopell_pop/libermentrout_kopell_pop.h",
    "src/sc_neurocore/accel/go/services/ermentrout_kopell_pop.go",
    "src/sc_neurocore/accel/go/services/ermentrout_kopell_pop_test.go",
    "src/sc_neurocore/accel/ermentrout_kopell_pop.py",
    "src/sc_neurocore/accel/julia/neurons/__init__.py",
    "src/sc_neurocore/accel/julia/neurons/ermentrout_kopell_pop.jl",
    "src/sc_neurocore/accel/mojo/ermentrout_kopell_pop/__init__.py",
    "src/sc_neurocore/accel/mojo/ermentrout_kopell_pop/ermentrout_kopell_pop.mojo",
    "src/sc_neurocore/accel/mojo/pixi.lock",
    "src/sc_neurocore/accel/mojo/pixi.toml",
    "src/sc_neurocore/accel/rust/safety/ermentrout_kopell_pop.rs",
    "src/sc_neurocore/neurons/model_descriptors/ErmentroutKopellPopulation.toml",
    "src/sc_neurocore/neurons/model_schemas/ermentrout_kopell_pop.json",
    "src/sc_neurocore/neurons/model_schemas/ermentrout_kopell_pop.toml",
    "src/sc_neurocore/neurons/models/ermentrout_kopell_pop.py",
    "src/sc_neurocore/neurons/reference_trace_data/ermentrout_kopell_pop_eq12_euler_doi.json",
    "tools/build_accel_backends.py",
    "tools/check_mojo_isa_baseline.py",
)

FloatArray: TypeAlias = npt.NDArray[np.float64]
BenchmarkResult: TypeAlias = dict[str, FloatArray | float]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _trace_sha256(result: BenchmarkResult) -> str:
    canonical = np.ascontiguousarray(
        np.column_stack([cast(FloatArray, result[key]) for key in TRACE_KEYS]),
        dtype="<f8",
    )
    return hashlib.sha256(canonical.tobytes()).hexdigest()


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


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPOSITORY.resolve()))
    except ValueError:
        return str(path.resolve())


def _binary_record(path: Path, *, display_path: str | None = None) -> dict[str, object]:
    resolved = path.resolve()
    return {
        "path": display_path or _display_path(resolved),
        "sha256": _sha256(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _binary_hashes() -> dict[str, dict[str, object]]:
    extension = importlib.import_module("sc_neurocore_engine.sc_neurocore_engine")
    extension_path = Path(str(extension.__file__))
    return {
        "rust_extension": _binary_record(
            extension_path,
            display_path=f"$WHEEL_SITE/sc_neurocore_engine/{extension_path.name}",
        ),
        "go_shared_library": _binary_record(
            REPOSITORY
            / "src/sc_neurocore/accel/go/ermentrout_kopell_pop/libermentrout_kopell_pop.so"
        ),
        "mojo_shared_library": _binary_record(
            REPOSITORY
            / "src/sc_neurocore/accel/mojo/ermentrout_kopell_pop/libermentrout_kopell_pop.so"
        ),
    }


def _drive(n_steps: int) -> FloatArray:
    index: FloatArray = np.arange(n_steps, dtype=np.float64)
    return 1.5 + 0.5 * np.sin(index * 0.037) + 0.25 * np.cos(index * 0.011)


def _probe_backend(backend: str) -> tuple[bool, str]:
    available = backend == "python" or backends.backend_available(backend)
    return available, "" if available else f"{backend} runtime or artefact unavailable"


def _run_backend(backend: str, n_steps: int) -> BenchmarkResult:
    return ErmentroutKopellPopulation(**INITIAL_STATE).simulate(_drive(n_steps), backend=backend)


def _measure_backend(
    backend: str,
    n_steps: int,
    repeats: int,
) -> tuple[list[int], BenchmarkResult]:
    _run_backend(backend, min(WARMUP_STEPS, n_steps))
    samples: list[int] = []
    result = _run_backend(backend, 0)
    for _ in range(repeats):
        start = time.perf_counter_ns()
        result = _run_backend(backend, n_steps)
        samples.append(time.perf_counter_ns() - start)
    return samples, result


def _tool_version(command: list[str]) -> str:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except OSError:
        return "unavailable"
    output = (completed.stdout or completed.stderr).strip().splitlines()
    return output[0] if output else f"exit {completed.returncode}"


def _go_binary_metadata() -> dict[str, object]:
    path = (
        REPOSITORY / "src/sc_neurocore/accel/go/ermentrout_kopell_pop/libermentrout_kopell_pop.so"
    )
    completed = subprocess.run(
        ["go", "version", "-m", str(path)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    fields: dict[str, str] = {}
    build_settings: dict[str, str] = {}
    lines = completed.stdout.splitlines()
    embedded_version = "unavailable"
    if lines:
        candidate = lines[0].rpartition(": ")[2].strip()
        if re.fullmatch(r"go\d+\.\d+(?:\.\d+)?", candidate):
            embedded_version = candidate
    for line in lines[1:]:
        columns = line.strip().split("\t")
        if len(columns) < 2:
            continue
        key, value = columns[0], columns[1]
        if key == "build" and "=" in value:
            setting, setting_value = value.split("=", 1)
            build_settings[setting] = setting_value
        elif key in {"go", "path", "mod"}:
            fields[key] = value
    return {
        "binary": _display_path(path),
        "go_version": embedded_version,
        "package": fields.get("path", "unavailable"),
        "module": fields.get("mod", "unavailable"),
        "cgo_enabled": build_settings.get("CGO_ENABLED", "unavailable"),
        "build_settings": build_settings,
    }


def _environment() -> dict[str, object]:
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
    juliacall = importlib.import_module("juliacall")
    mojo_manifest = REPOSITORY / "src/sc_neurocore/accel/mojo/pixi.toml"
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "affinity": affinity,
        "load_average": list(os.getloadavg()) if hasattr(os, "getloadavg") else [],
        "rustc": _tool_version(["rustc", "--version"]),
        "julia_runtime": str(juliacall.Main.seval("string(VERSION)")),
        "julia_cli": _tool_version(["julia", "--version"]),
        "go_binary": _go_binary_metadata(),
        "go_cli": _tool_version(["go", "version"]),
        "mojo_pixi": _tool_version(
            [
                "pixi",
                "run",
                "--manifest-path",
                str(mojo_manifest),
                "mojo",
                "--version",
            ]
        ),
        "mojo_cli": _tool_version(["mojo", "--version"]),
    }


def _verify_rust_safety() -> dict[str, object]:
    source = REPOSITORY / "src/sc_neurocore/accel/rust/safety/ermentrout_kopell_pop.rs"
    with tempfile.TemporaryDirectory(prefix="mpr-rust-") as directory:
        binary = Path(directory) / "ermentrout_kopell_pop_tests"
        command = ["rustc", "--edition", "2021", "--test", str(source), "-o", str(binary)]
        compiled = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if compiled.returncode != 0:
            executed = compiled
        else:
            executed = subprocess.run(
                [str(binary)],
                check=False,
                capture_output=True,
                text=True,
                timeout=60,
            )
    recorded_command = [
        "rustc",
        "--edition",
        "2021",
        "--test",
        _display_path(source),
        "-o",
        "<temporary>/ermentrout_kopell_pop_tests",
    ]
    return {
        "command": recorded_command,
        "passed": executed.returncode == 0,
        "returncode": executed.returncode,
        "output_tail": (executed.stdout + executed.stderr).splitlines()[-20:],
    }


def _backend_record(
    backend: str,
    samples: list[int],
    result: BenchmarkResult,
    reference: BenchmarkResult,
    n_steps: int,
) -> dict[str, object]:
    tolerance = PARITY_ATOL[backend]
    differences = [
        np.abs(cast(FloatArray, result[key]) - cast(FloatArray, reference[key]))
        for key in TRACE_KEYS
    ]
    mismatch_count = sum(
        int(np.count_nonzero(difference > tolerance)) for difference in differences
    )
    max_difference = max(
        (float(difference.max(initial=0.0)) for difference in differences),
        default=0.0,
    )
    final_difference = max(
        (abs(float(result[key]) - float(reference[key])) for key in FINAL_KEYS),
        default=0.0,
    )
    median_ns = float(statistics.median(samples))
    return {
        "available": True,
        "used": True,
        "samples_ns": samples,
        "median_call_ns": median_ns,
        "median_call_ms": median_ns / 1_000_000.0,
        "min_call_ns": min(samples),
        "max_call_ns": max(samples),
        "median_ns_per_step": median_ns / n_steps if n_steps else 0.0,
        "trace_sha256": _trace_sha256(result),
        "trace_matches_python": mismatch_count == 0,
        "trace_mismatch_count": mismatch_count,
        "parity_max_abs_diff": max_difference,
        "parity_atol": tolerance,
        "final_state": {key.removesuffix("_final"): float(result[key]) for key in FINAL_KEYS},
        "final_state_max_abs_diff": final_difference,
        "final_state_matches_python": final_difference <= tolerance,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--steps", type=int, default=N_STEPS)
    parser.add_argument("--repeats", type=int, default=N_REPEATS)
    parser.add_argument("--allow-unpinned", action="store_true")
    parser.add_argument("--allow-unavailable-backends", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.steps <= 0 or args.repeats <= 0:
        raise ValueError("steps and repeats must be positive")
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
    if len(affinity) != 1 and not args.allow_unpinned:
        print("benchmark requires one pinned logical CPU or --allow-unpinned", flush=True)
        return 2
    probes = {backend: _probe_backend(backend) for backend in BACKENDS}
    missing = [backend for backend, (available, _) in probes.items() if not available]
    if missing and not args.allow_unavailable_backends:
        print(f"required backends unavailable: {', '.join(missing)}", flush=True)
        return 2
    rust_safety = _verify_rust_safety()
    if not rust_safety["passed"]:
        print("standalone Rust-safety verification failed", flush=True)
        return 4

    measured: dict[str, dict[str, object]] = {}
    reference: BenchmarkResult | None = None
    parity_failed = False
    for backend in BACKENDS:
        available, reason = probes[backend]
        if not available:
            measured[backend] = {"available": False, "used": False, "unavailable_reason": reason}
            continue
        samples, result = _measure_backend(backend, args.steps, args.repeats)
        if backend == "python":
            reference = result
        if reference is None:
            raise RuntimeError("Python reference must be measured first")
        record = _backend_record(backend, samples, result, reference, args.steps)
        measured[backend] = record
        parity_failed |= not bool(record["trace_matches_python"])
        parity_failed |= not bool(record["final_state_matches_python"])
    if parity_failed:
        print("runtime parity exceeded the declared tolerance", flush=True)
        return 3

    native_order = sorted(
        (backend for backend in BACKENDS if backend != "python" and measured[backend]["used"]),
        key=lambda name: float(cast(SupportsFloat, measured[name]["median_call_ns"])),
    )
    report: dict[str, Any] = {
        "schema_version": "sc-neurocore.polyglot-benchmark.v1",
        "benchmark": "ErmentroutKopellPopulation five-runtime equation-(12) batch",
        "kernel": KERNEL,
        "timestamp_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "command": (
            "taskset -c <cpu> env PYTHONPATH=$WHEEL_SITE:src:. .venv/bin/python "
            "benchmarks/bench_ermentrout_kopell_pop.py --json benchmarks/results/bench_ermentrout_kopell_pop.json"
        ),
        "evidence_class": "local_regression_single_cpu_affinity_non_exclusive",
        "production_speed_claim": False,
        "hardware_measurement_claimed": False,
        "parity_contract": {
            "observable": "two complete float64 traces and two final states",
            "comparison": {name: PARITY_ATOL[name] for name in BACKENDS},
            "reason": "independent compiler evaluation orders may differ by bounded ulps",
        },
        "workload": {
            "n_steps": args.steps,
            "repeats": args.repeats,
            "warmup_steps": min(WARMUP_STEPS, args.steps),
            "input_formula": "1.5 + 0.5*sin(step*0.037) + 0.25*cos(step*0.011)",
            "initial_state": INITIAL_STATE,
        },
        "measured_order": list(BACKENDS),
        "lowest_median_native_backend": native_order[0] if native_order else None,
        "recommended_auto_backend": native_order[0] if native_order else "python",
        "auto_backend_order": [*native_order, "python"],
        "auto_backend_selection_basis": (
            "same-host measured warm batch order; non-exclusive timings remain diagnostic"
        ),
        "backends": measured,
        "verification": {"rust_safety": rust_safety},
        "source_hashes": _source_hashes(),
        "binary_hashes": _binary_hashes(),
        "environment": _environment(),
        "meta": {
            "cpu": current_cpu(),
            "single_cpu_pinned": len(affinity) == 1,
            "exclusive_cpu_isolation_claimed": False,
            "runtime_cpuset_shield_claimed": False,
        },
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
