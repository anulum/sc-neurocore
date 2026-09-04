# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source-bound five-runtime Wong-Wang benchmark

"""Measure and fail-closed verify the complete Wong-Wang runtime set."""

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
from typing import Any, SupportsFloat, TypeAlias, cast

import numpy as np
import numpy.typing as npt

from sc_neurocore.accel import wong_wang as backends
from sc_neurocore.accel.backend_selection import current_cpu
from sc_neurocore.neurons.models.wong_wang import WongWangUnit

REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPOSITORY / "benchmarks/results/bench_wong_wang.json"
KERNEL = backends.KERNEL
BACKENDS = ("python", "rust", "julia", "go", "mojo")
N_STEPS = 100_000
N_REPEATS = 5
WARMUP_STEPS = 1_000
PARITY_ATOL = backends.PARITY_ATOL
ATOL = max(PARITY_ATOL.values())
TRACE_KEYS = ("s1", "s2", "noise1", "noise2", "r1", "r2")
FINAL_KEYS = ("s1_final", "s2_final", "noise1_final", "noise2_final")
INITIAL_STATE = {
    "s1": 0.24,
    "s2": 0.11,
    "noise1": 0.01,
    "noise2": -0.02,
    "tau_s": 0.12,
    "tau_ampa": 0.003,
    "gamma": 0.7,
    "j_n": 0.28,
    "j_cross": 0.06,
    "i_0": 0.31,
    "sigma": 0.015,
    "dt": 0.0002,
}
SOURCE_PATHS = (
    "benchmarks/bench_wong_wang.py",
    "bridge/sc_neurocore_engine/__init__.py",
    "engine/src/bindings/wong_wang.rs",
    "engine/src/neurons/special.rs",
    "engine/src/wong_wang.rs",
    "src/sc_neurocore/accel/go/services/wong_wang.go",
    "src/sc_neurocore/accel/go/services/wong_wang_test.go",
    "src/sc_neurocore/accel/go/wong_wang/__init__.py",
    "src/sc_neurocore/accel/go/wong_wang/wong_wang.go",
    "src/sc_neurocore/accel/julia/neurons/__init__.py",
    "src/sc_neurocore/accel/julia/neurons/wong_wang.jl",
    "src/sc_neurocore/accel/mojo/wong_wang/__init__.py",
    "src/sc_neurocore/accel/mojo/wong_wang/wong_wang.mojo",
    "src/sc_neurocore/accel/rust/safety/wong_wang.rs",
    "src/sc_neurocore/accel/wong_wang.py",
    "src/sc_neurocore/neurons/model_descriptors/WongWangUnit.toml",
    "src/sc_neurocore/neurons/model_schemas/wong_wang.json",
    "src/sc_neurocore/neurons/model_schemas/wong_wang.toml",
    "src/sc_neurocore/neurons/models/wong_wang.py",
    "src/sc_neurocore/neurons/reference_trace_data/wong_wang_appendix_euler_ou_doi.json",
)

FloatArray: TypeAlias = npt.NDArray[np.float64]
WongWangResult: TypeAlias = dict[str, FloatArray | float]


def _sha256(path: Path) -> str:
    """Return the binary SHA-256 digest for one file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _trace_sha256(result: WongWangResult) -> str:
    """Hash canonical interleaved little-endian six-observable trajectories."""
    canonical = np.ascontiguousarray(
        np.column_stack([cast(FloatArray, result[key]) for key in TRACE_KEYS]),
        dtype="<f8",
    )
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def _source_hashes() -> dict[str, object]:
    """Bind sources with flat paths plus gate-addressable suffix aliases."""
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
    """Prefer a repository-relative artefact path when possible."""
    try:
        return str(path.resolve().relative_to(REPOSITORY.resolve()))
    except ValueError:
        return str(path.resolve())


def _binary_record(path: Path) -> dict[str, object]:
    """Return the digest, size, and location of one loaded native object."""
    resolved = path.resolve()
    return {
        "path": _display_path(resolved),
        "sha256": _sha256(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _binary_hashes() -> dict[str, dict[str, object]]:
    """Bind the measured Rust, Go, and Mojo machine-code artefacts."""
    extension = importlib.import_module("sc_neurocore_engine.sc_neurocore_engine")
    return {
        "rust_extension": _binary_record(Path(str(extension.__file__))),
        "go_shared_library": _binary_record(
            REPOSITORY / "src/sc_neurocore/accel/go/wong_wang/libwong_wang.so"
        ),
        "mojo_shared_library": _binary_record(
            REPOSITORY / "src/sc_neurocore/accel/mojo/wong_wang/libwong_wang.so"
        ),
    }


def _inputs(n_steps: int) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return varied deterministic currents and explicit normal samples."""
    index: FloatArray = np.arange(n_steps, dtype=np.float64)
    return (
        0.02 + 0.01 * np.sin(index * 0.07),
        -0.01 + 0.008 * np.cos(index * 0.11),
        np.sin(np.arange(2 * n_steps, dtype=np.float64) * 0.17),
    )


def _probe_backend(backend: str) -> tuple[bool, str]:
    """Report one public runtime without substituting a surrogate."""
    if backend == "python":
        return True, ""
    available = backends.backend_available(backend)
    return available, "" if available else f"{backend} runtime or artefact unavailable"


def _run_backend(backend: str, n_steps: int) -> WongWangResult:
    """Run one fresh complete batch through the public model dispatcher."""
    stim1, stim2, xi = _inputs(n_steps)
    return WongWangUnit(**INITIAL_STATE).simulate(stim1, stim2, xi, backend=backend)


def _measure_backend(
    backend: str,
    n_steps: int,
    repeats: int,
) -> tuple[list[int], WongWangResult]:
    """Warm then measure one runtime through its public dispatcher."""
    _run_backend(backend, min(WARMUP_STEPS, n_steps))
    samples: list[int] = []
    result = _run_backend(backend, 0)
    for _ in range(repeats):
        start = time.perf_counter_ns()
        result = _run_backend(backend, n_steps)
        samples.append(time.perf_counter_ns() - start)
    return samples, result


def _tool_version(command: list[str]) -> str:
    """Return the first non-empty version line or an explicit fallback."""
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


def _environment() -> dict[str, object]:
    """Record enough host context to prevent production-speed overclaiming."""
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "affinity": affinity,
        "load_average": list(os.getloadavg()) if hasattr(os, "getloadavg") else [],
        "rustc": _tool_version(["rustc", "--version"]),
        "go": _tool_version(["go", "version"]),
        "julia": _tool_version(["julia", "--version"]),
        "mojo": _tool_version(["mojo", "--version"]),
    }


def _verify_rust_safety() -> dict[str, object]:
    """Compile and execute the standalone Rust-safety module tests."""
    source = REPOSITORY / "src/sc_neurocore/accel/rust/safety/wong_wang.rs"
    with tempfile.TemporaryDirectory(prefix="wong-wang-rust-") as directory:
        binary = Path(directory) / "wong_wang_tests"
        command = ["rustc", "--edition", "2021", "--test", str(source), "-o", str(binary)]
        display_command = [
            "rustc",
            "--edition",
            "2021",
            "--test",
            _display_path(source),
            "-o",
            "<temporary>/wong_wang_tests",
        ]
        try:
            compiled = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=120,
            )
            if compiled.returncode != 0:
                return {
                    "command": display_command,
                    "passed": False,
                    "returncode": compiled.returncode,
                    "output_tail": (compiled.stdout + compiled.stderr).splitlines()[-20:],
                }
            executed = subprocess.run(
                [str(binary)],
                check=False,
                capture_output=True,
                text=True,
                timeout=60,
            )
        except OSError as exc:
            return {
                "command": display_command,
                "passed": False,
                "returncode": -1,
                "output_tail": [str(exc)],
            }
    return {
        "command": display_command,
        "passed": executed.returncode == 0,
        "returncode": executed.returncode,
        "output_tail": (executed.stdout + executed.stderr).splitlines()[-20:],
    }


def _backend_record(
    backend: str,
    samples: list[int],
    result: WongWangResult,
    reference: WongWangResult,
    n_steps: int,
) -> dict[str, object]:
    """Summarise timing and bounded numerical parity for one runtime."""
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
    final_differences = {key: abs(float(result[key]) - float(reference[key])) for key in FINAL_KEYS}
    final_difference = max(final_differences.values(), default=0.0)
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
    """Run the benchmark, reject incomplete parity, and write JSON evidence."""
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
    measured_order: list[str] = []
    reference: WongWangResult | None = None
    parity_failed = False
    for backend in BACKENDS:
        available, reason = probes[backend]
        if not available:
            measured[backend] = {
                "available": False,
                "used": False,
                "unavailable_reason": reason,
            }
            continue
        samples, result = _measure_backend(backend, args.steps, args.repeats)
        if backend == "python":
            reference = result
        if reference is None:
            raise RuntimeError("Python reference must be measured first")
        record = _backend_record(backend, samples, result, reference, args.steps)
        measured[backend] = record
        measured_order.append(backend)
        parity_failed |= not bool(record["trace_matches_python"])
        parity_failed |= not bool(record["final_state_matches_python"])

    native_order = sorted(
        (backend for backend in measured_order if backend != "python"),
        key=lambda name: float(cast(SupportsFloat, measured[name]["median_call_ns"])),
    )
    report: dict[str, Any] = {
        "schema_version": "sc-neurocore.polyglot-benchmark.v1",
        "benchmark": "WongWangUnit five-runtime Euler/OU batch",
        "kernel": KERNEL,
        "timestamp_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "command": (
            "taskset -c <cpu> env PYTHONPATH=src:. .venv/bin/python "
            "benchmarks/bench_wong_wang.py --json "
            "benchmarks/results/bench_wong_wang.json"
        ),
        "evidence_class": "local_regression_single_cpu_affinity_non_exclusive",
        "production_speed_claim": False,
        "hardware_measurement_claimed": False,
        "parity_contract": {
            "observable": "six complete post-step/rate float64 traces and four final states",
            "comparison": {
                "python": "reference",
                "rust": "absolute tolerance 1.0e-12",
                "julia": "absolute tolerance 1.0e-12",
                "go": "absolute tolerance 1.0e-12",
                "mojo": "absolute tolerance 1.0e-9",
            },
            "reason": "transcendental library implementations may differ by bounded ulps",
        },
        "workload": {
            "n_steps": args.steps,
            "repeats": args.repeats,
            "warmup_steps": min(WARMUP_STEPS, args.steps),
            "input_formula": {
                "stim1": "0.02 + 0.01*sin(step*0.07)",
                "stim2": "-0.01 + 0.008*cos(step*0.11)",
                "xi": "sin(sample_index*0.17)",
            },
            "initial_state": INITIAL_STATE,
        },
        "measured_order": measured_order,
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
    if parity_failed:
        print("Wong-Wang parity failed; benchmark evidence was not written")
        return 3
    rendered = json.dumps(report, indent=2, sort_keys=True)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(rendered + "\n", encoding="utf-8")
    print(f"Wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
