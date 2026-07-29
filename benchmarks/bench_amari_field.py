# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — source-bound five-runtime Amari field benchmark

"""Measure and fail-closed verify complete vector receipts on all five lanes."""

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
import numpy as np
import numpy.typing as npt

from sc_neurocore.accel import amari_field as backends

REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPOSITORY / "benchmarks/results/bench_amari_field.json"
BACKENDS = ("python", "rust", "julia", "go", "mojo")
KERNEL = backends.KERNEL
PARITY_ATOL = backends.PARITY_ATOL
N_SITES = 16
N_STEPS = 20_000
N_REPEATS = 3
SOURCE_PATHS = (
    "benchmarks/bench_amari_field.py",
    "engine/src/bindings/rate/amari_neural_field.rs",
    "engine/src/neurons/rate/amari_neural_field.rs",
    "hdl/formal/catalogue/sc_amari_field.v",
    "src/sc_neurocore/accel/amari_field.py",
    "src/sc_neurocore/accel/go/amari_field/amari_field.go",
    "src/sc_neurocore/accel/go/services/amari_field.go",
    "src/sc_neurocore/accel/julia/neurons/amari_field.jl",
    "src/sc_neurocore/accel/mojo/amari_field/amari_field.mojo",
    "src/sc_neurocore/accel/rust/safety/amari_field.rs",
    "src/sc_neurocore/neurons/model_descriptors/AmariNeuralField.toml",
    "src/sc_neurocore/neurons/model_schemas/amari_field.json",
    "src/sc_neurocore/neurons/model_schemas/amari_field.toml",
    "src/sc_neurocore/neurons/models/amari_field.py",
    "src/sc_neurocore/neurons/reference_trace_data/amari_field_doi.json",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_hashes() -> dict[str, str]:
    """Bind every maintained source surface used by the measurement."""
    return {relative: _sha256(REPOSITORY / relative) for relative in SOURCE_PATHS}


def _binary_record(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    try:
        display = str(resolved.relative_to(REPOSITORY.resolve()))
    except ValueError:
        display = str(resolved)
    return {"path": display, "sha256": _sha256(resolved), "size_bytes": resolved.stat().st_size}


def _binary_hashes() -> dict[str, dict[str, object]]:
    """Bind the exact Rust extension and Go/Mojo shared objects executed."""
    extension = importlib.import_module("sc_neurocore_engine.sc_neurocore_engine")
    return {
        "rust_extension": _binary_record(Path(str(extension.__file__))),
        "go_shared_library": _binary_record(
            REPOSITORY / "src/sc_neurocore/accel/go/amari_field/libamari_field.so"
        ),
        "mojo_shared_library": _binary_record(
            REPOSITORY / "src/sc_neurocore/accel/mojo/amari_field/libamari_field.so"
        ),
    }


def _initial_state() -> npt.NDArray[np.float64]:
    return np.linspace(-0.25, 0.25, N_SITES, dtype=np.float64)


def _drives(steps: int) -> npt.NDArray[np.float64]:
    time_index = np.arange(steps, dtype=np.float64)[:, None]
    sites = np.arange(N_SITES, dtype=np.float64)[None, :]
    return np.ascontiguousarray(
        0.08 * np.sin(time_index * 0.013 + sites * 0.21)
        + 0.025 * np.cos(time_index * 0.031 - sites * 0.17)
        - 0.02
    )


def _run_backend(backend: str, steps: int) -> backends.AmariResult:
    return backends.simulate_amari_field(_initial_state(), currents=_drives(steps), backend=backend)


def _trace_hash(result: backends.AmariResult) -> str:
    values = np.column_stack((result["states"], result["mean_rates"]))
    return hashlib.sha256(np.ascontiguousarray(values, dtype="<f8").tobytes()).hexdigest()


def _measure(backend: str, steps: int, repeats: int) -> tuple[list[int], backends.AmariResult]:
    _run_backend(backend, min(128, steps))
    samples = []
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
    source = REPOSITORY / "src/sc_neurocore/accel/rust/safety/amari_field.rs"
    with tempfile.TemporaryDirectory(prefix="amari-safety-") as directory:
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
            [str(binary)], check=False, capture_output=True, text=True, timeout=30
        )
        return {
            "passed": executed.returncode == 0,
            "returncode": executed.returncode,
            "output_tail": (executed.stdout + executed.stderr).splitlines()[-20:],
        }


def build_payload(steps: int, repeats: int) -> tuple[dict[str, object], bool]:
    """Measure every backend and return evidence plus its aggregate verdict."""
    reference: backends.AmariResult | None = None
    rows: dict[str, dict[str, object]] = {}
    order: list[tuple[int, str]] = []
    passed = True
    for backend in BACKENDS:
        available = backends.backend_available(backend)
        if not available:
            rows[backend] = {"available": False, "used": False, "reason": "runtime unavailable"}
            passed = False
            continue
        samples, result = _measure(backend, steps, repeats)
        if reference is None:
            reference = result
        tolerance = PARITY_ATOL[backend]
        state_gap = float(np.max(np.abs(result["states"] - reference["states"]), initial=0.0))
        final_gap = float(
            np.max(np.abs(result["final_state"] - reference["final_state"]), initial=0.0)
        )
        rates_exact = bool(np.array_equal(result["mean_rates"], reference["mean_rates"]))
        parity = state_gap <= tolerance and final_gap <= tolerance and rates_exact
        median = int(statistics.median(samples))
        order.append((median, backend))
        rows[backend] = {
            "available": True,
            "used": True,
            "samples_ns": samples,
            "median_ns": median,
            "ns_per_site_step": median / max(1, steps * N_SITES),
            "state_max_abs_diff": state_gap,
            "final_state_max_abs_diff": final_gap,
            "mean_rates_exact": rates_exact,
            "parity_tolerance": tolerance,
            "trace_matches_python": parity,
            "trace_sha256": _trace_hash(result),
            "active_site_rate_sum": float(np.sum(result["mean_rates"])),
        }
        passed = passed and parity
    safety = _verify_rust_safety()
    passed = passed and bool(safety["passed"])
    payload: dict[str, object] = {
        "schema_version": "sc-neurocore.polyglot-benchmark.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "kernel": KERNEL,
        "model": "AmariNeuralField",
        "evidence_class": "local_regression",
        "production_speed_claim": False,
        "hardware_measurement_claimed": False,
        "continuous_space_convergence_claimed": False,
        "configuration": {
            "sites": N_SITES,
            "steps": steps,
            "repeats": repeats,
            "tau": 10.0,
            "dx": 0.5,
            "dt": 0.5,
        },
        "meta": {
            "single_cpu_pinned": len(os.sched_getaffinity(0)) == 1,
            "exclusive_cpu_isolation_claimed": False,
        },
        "environment": _environment(),
        "source_hashes": _source_hashes(),
        "binary_hashes": _binary_hashes(),
        "verification": {"rust_safety": safety},
        "backends": rows,
        "measured_order": [name for _, name in sorted(order)],
        "passed": passed,
    }
    return payload, passed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--steps", type=int, default=N_STEPS)
    parser.add_argument("--repeats", type=int, default=N_REPEATS)
    args = parser.parse_args(argv)
    if args.steps < 1 or args.repeats < 1:
        return 2
    if hasattr(os, "sched_getaffinity") and len(os.sched_getaffinity(0)) != 1:
        print("benchmark requires taskset pinning to one logical CPU")
        return 2
    payload, passed = build_payload(args.steps, args.repeats)
    if not passed:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 3
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
