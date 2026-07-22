# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source-bound five-backend threshold-linear rate benchmark

"""Measure and fail-closed verify the complete threshold-linear backend set."""

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
from typing import Any, SupportsFloat, cast

import numpy as np
import numpy.typing as npt

from sc_neurocore.accel import threshold_linear_rate as backends
from sc_neurocore.accel.backend_selection import current_cpu
from sc_neurocore.neurons.models.threshold_linear_rate import ThresholdLinearRateNeuron

REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    REPOSITORY / "benchmarks/results/local_python_2026-07-14_threshold_linear_rate.json"
)
KERNEL = backends.KERNEL
BACKENDS = ("python", "rust", "julia", "go", "mojo")
N_STEPS = 200_000
N_REPEATS = 5
WARMUP_STEPS = 1_000
CURRENT = 3.0
INITIAL_STATE = {"r": 0.25, "theta": 1.5, "gain": 2.0}
SOURCE_PATHS = (
    "benchmarks/bench_model_threshold_linear_rate.py",
    "bridge/sc_neurocore_engine/__init__.py",
    "engine/src/bindings/threshold_linear_rate.rs",
    "engine/src/neurons/rate.rs",
    "engine/src/pyo3_neurons.rs",
    "src/sc_neurocore/accel/threshold_linear_rate.py",
    "src/sc_neurocore/accel/go/neurons/threshold_linear_rate/libthreshold_linear_rate.h",
    "src/sc_neurocore/accel/go/neurons/threshold_linear_rate/threshold_linear_rate.go",
    "src/sc_neurocore/accel/go/services/threshold_linear_rate.go",
    "src/sc_neurocore/accel/go/services/threshold_linear_rate_test.go",
    "src/sc_neurocore/accel/julia/neurons/threshold_linear_rate.jl",
    "src/sc_neurocore/accel/mojo/kernels/threshold_linear_rate.mojo",
    "src/sc_neurocore/accel/rust/safety/threshold_linear_rate.rs",
    "src/sc_neurocore/neurons/model_descriptors/ThresholdLinearRateNeuron.toml",
    "src/sc_neurocore/neurons/model_schemas/threshold_linear_rate.toml",
    "src/sc_neurocore/neurons/models/threshold_linear_rate.py",
)


def _sha256(path: Path) -> str:
    """Return the binary SHA-256 digest for one file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _trace_sha256(trace: npt.NDArray[np.float64]) -> str:
    """Hash a canonical little-endian float64 trajectory."""
    canonical = np.ascontiguousarray(trace, dtype="<f8")
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
    """Prefer a repository-relative artifact path when possible."""
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
    """Bind the measured Rust, Go, and Mojo machine-code artifacts."""
    extension = importlib.import_module("sc_neurocore_engine.sc_neurocore_engine")
    return {
        "rust_extension": _binary_record(Path(str(extension.__file__))),
        "go_shared_library": _binary_record(
            REPOSITORY
            / "src/sc_neurocore/accel/go/neurons/threshold_linear_rate/libthreshold_linear_rate.so"
        ),
        "mojo_shared_library": _binary_record(
            REPOSITORY / "src/sc_neurocore/accel/mojo/kernels/libthreshold_linear_rate.so"
        ),
    }


def _probe_backend(backend: str) -> tuple[bool, str]:
    """Report one public runtime without substituting a surrogate."""
    if backend == "python":
        return True, ""
    available = backends.backend_available(backend)
    return available, "" if available else f"{backend} runtime or artifact unavailable"


def _run_backend(backend: str, n_steps: int) -> tuple[npt.NDArray[np.float64], float]:
    """Run one fresh full-contract batch and return its trace and final rate."""
    neuron = ThresholdLinearRateNeuron(**INITIAL_STATE)
    trace = neuron.simulate(n_steps, CURRENT, backend=backend)
    return trace, neuron.r


def _measure_backend(
    backend: str,
    n_steps: int,
    repeats: int,
) -> tuple[list[int], npt.NDArray[np.float64], float]:
    """Warm then measure one backend through its public dispatcher."""
    _run_backend(backend, min(WARMUP_STEPS, n_steps))
    samples: list[int] = []
    trace: npt.NDArray[np.float64] = np.empty(0, dtype=np.float64)
    final_rate = INITIAL_STATE["r"]
    for _ in range(repeats):
        start = time.perf_counter_ns()
        trace, final_rate = _run_backend(backend, n_steps)
        samples.append(time.perf_counter_ns() - start)
    return samples, trace, final_rate


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
    source = REPOSITORY / "src/sc_neurocore/accel/rust/safety/threshold_linear_rate.rs"
    with tempfile.TemporaryDirectory(prefix="threshold-linear-rate-rust-") as directory:
        binary = Path(directory) / "threshold_linear_rate_tests"
        command = ["rustc", "--edition", "2021", "--test", str(source), "-o", str(binary)]
        display_command = [
            "rustc",
            "--edition",
            "2021",
            "--test",
            _display_path(source),
            "-o",
            "<temporary>/threshold_linear_rate_tests",
        ]
        try:
            compiled = subprocess.run(
                command, check=False, capture_output=True, text=True, timeout=120
            )
            if compiled.returncode != 0:
                return {
                    "command": display_command,
                    "passed": False,
                    "returncode": compiled.returncode,
                    "output_tail": (compiled.stdout + compiled.stderr).splitlines()[-20:],
                }
            executed = subprocess.run(
                [str(binary)], check=False, capture_output=True, text=True, timeout=60
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
    samples: list[int],
    trace: npt.NDArray[np.float64],
    final_rate: float,
    reference: npt.NDArray[np.float64],
    reference_final: float,
    n_steps: int,
) -> dict[str, object]:
    """Summarise timing and exact numerical parity for one runtime."""
    differences = np.abs(trace - reference)
    mismatch_count = int(np.count_nonzero(trace != reference))
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
        "trace_sha256": _trace_sha256(trace),
        "trace_matches_python": mismatch_count == 0,
        "trace_mismatch_count": mismatch_count,
        "parity_max_abs_diff": float(differences.max(initial=0.0)),
        "final_rate": final_rate,
        "final_rate_matches_python": final_rate == reference_final,
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
    reference: npt.NDArray[np.float64] = np.empty(0, dtype=np.float64)
    reference_final = INITIAL_STATE["r"]
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
        samples, trace, final_rate = _measure_backend(backend, args.steps, args.repeats)
        if backend == "python":
            reference = trace
            reference_final = final_rate
        elif reference.shape != trace.shape:
            raise RuntimeError("Python reference must be measured first")
        record = _backend_record(samples, trace, final_rate, reference, reference_final, args.steps)
        measured[backend] = record
        measured_order.append(backend)
        parity_failed |= not bool(record["trace_matches_python"])
        parity_failed |= not bool(record["final_rate_matches_python"])

    native_order = sorted(
        (backend for backend in measured_order if backend != "python"),
        key=lambda name: float(cast(SupportsFloat, measured[name]["median_call_ns"])),
    )
    report: dict[str, Any] = {
        "schema_version": "sc-neurocore.polyglot-benchmark.v1",
        "benchmark": "ThresholdLinearRateNeuron five-backend algebraic batch",
        "kernel": KERNEL,
        "timestamp_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "command": (
            "taskset -c <cpu> env PYTHONPATH=bridge:src:. .venv/bin/python "
            "benchmarks/bench_model_threshold_linear_rate.py --json "
            "benchmarks/results/local_python_2026-07-14_threshold_linear_rate.json"
        ),
        "evidence_class": "local_regression_single_cpu_affinity_non_exclusive",
        "production_speed_claim": False,
        "hardware_measurement_claimed": False,
        "parity_contract": {
            "observable": "complete post-evaluation float64 rate trace and final cached output",
            "comparison": "bit-exact",
            "reason": "the maintained transfer uses only subtraction, maximum, and multiplication",
        },
        "workload": {
            "n_steps": args.steps,
            "repeats": args.repeats,
            "warmup_steps": min(WARMUP_STEPS, args.steps),
            "current": CURRENT,
            "initial_state": INITIAL_STATE,
        },
        "measured_order": measured_order,
        "fastest_measured_native_backend": native_order[0] if native_order else None,
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
        print("Threshold-linear parity failed; benchmark evidence was not written")
        return 3
    rendered = json.dumps(report, indent=2, sort_keys=True)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(rendered + "\n", encoding="utf-8")
    print(f"Wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
