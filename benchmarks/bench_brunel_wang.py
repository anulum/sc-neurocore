# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — source/binary-bound five-runtime Brunel-Wang benchmark

"""Measure complete Brunel-Wang trajectories without production-speed claims."""

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

from sc_neurocore.accel import brunel_wang as backends
from sc_neurocore.neurons.models.brunel_wang import BrunelWangNeuron

REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPOSITORY / "benchmarks/results/bench_brunel_wang.json"
BACKENDS = ("python", "rust", "julia", "go", "mojo")
KERNEL = backends.KERNEL
PARITY_ATOL = backends.PARITY_ATOL
N_STEPS = 200_000
N_REPEATS = 3
SOURCE_PATHS = (
    "benchmarks/bench_brunel_wang.py",
    "bridge/sc_neurocore_engine/__init__.py",
    "engine/src/bindings/population/brunel_wang.rs",
    "engine/src/neurons/simple_spiking/brunel_wang.rs",
    "hdl/formal/catalogue/sc_brunel_wang.v",
    "src/sc_neurocore/accel/brunel_wang.py",
    "src/sc_neurocore/accel/go/brunel_wang/brunel_wang.go",
    "src/sc_neurocore/accel/go/services/brunel_wang.go",
    "src/sc_neurocore/accel/julia/neurons/brunel_wang.jl",
    "src/sc_neurocore/accel/mojo/brunel_wang/brunel_wang.mojo",
    "src/sc_neurocore/accel/rust/safety/brunel_wang.rs",
    "src/sc_neurocore/neurons/model_descriptors/BrunelWangNeuron.toml",
    "src/sc_neurocore/neurons/model_schemas/brunel_wang.json",
    "src/sc_neurocore/neurons/model_schemas/brunel_wang.toml",
    "src/sc_neurocore/neurons/models/brunel_wang.py",
    "src/sc_neurocore/neurons/reference_trace_data/brunel_wang_2001_pyramidal.json",
)


def _sha256(path: Path) -> str:
    """Return one binary SHA-256 digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_hashes() -> dict[str, str]:
    """Bind every maintained implementation surface used by this result."""
    return {relative: _sha256(REPOSITORY / relative) for relative in SOURCE_PATHS}


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPOSITORY.resolve()))
    except ValueError:
        return str(path.resolve())


def _binary_record(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    return {
        "path": _display_path(resolved),
        "sha256": _sha256(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _binary_hashes() -> dict[str, dict[str, object]]:
    """Bind the exact Rust, Go, and Mojo objects executed."""
    extension = importlib.import_module("sc_neurocore_engine.sc_neurocore_engine")
    return {
        "rust_extension": _binary_record(Path(str(extension.__file__))),
        "go_shared_library": _binary_record(
            REPOSITORY / "src/sc_neurocore/accel/go/brunel_wang/libbrunel_wang.so"
        ),
        "mojo_shared_library": _binary_record(
            REPOSITORY / "src/sc_neurocore/accel/mojo/brunel_wang/libbrunel_wang.so"
        ),
    }


def _gates(steps: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    index = np.arange(steps, dtype=np.float64)
    return (
        0.053 + 0.018 * np.sin(index * 0.071),
        0.17 + 0.05 * np.cos(index * 0.053),
        0.12 + 0.04 * np.sin(index * 0.037 + 0.2),
        0.05 + 0.02 * np.cos(index * 0.089),
    )


def _run_backend(backend: str, steps: int) -> backends.BrunelWangResult:
    """Run one fresh, complete batch through the public model dispatcher."""
    return BrunelWangNeuron().simulate(*_gates(steps), backend=backend)


def _trace_hash(result: backends.BrunelWangResult) -> str:
    values = np.column_stack((result["voltages"], result["refractory"], result["events"]))
    return hashlib.sha256(np.ascontiguousarray(values, dtype="<f8").tobytes()).hexdigest()


def _measure(backend: str, steps: int, repeats: int) -> tuple[list[int], backends.BrunelWangResult]:
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
    source = REPOSITORY / "src/sc_neurocore/accel/rust/safety/brunel_wang.rs"
    with tempfile.TemporaryDirectory(prefix="brunel-wang-safety-") as directory:
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
    """Measure all five real lanes and return their aggregate verdict."""
    reference: backends.BrunelWangResult | None = None
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
        voltages = cast(np.ndarray, result["voltages"])
        refractory = cast(np.ndarray, result["refractory"])
        ref_voltages = cast(np.ndarray, reference["voltages"])
        ref_refractory = cast(np.ndarray, reference["refractory"])
        event_exact = bool(np.array_equal(result["events"], reference["events"]))
        max_gap = max(
            float(np.max(np.abs(voltages - ref_voltages), initial=0.0)),
            float(np.max(np.abs(refractory - ref_refractory), initial=0.0)),
            abs(float(result["v_final"]) - float(reference["v_final"])),
            abs(float(result["ref_final"]) - float(reference["ref_final"])),
        )
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
            "event_count": int(np.sum(cast(np.ndarray, result["events"]))),
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
        "model": "BrunelWangNeuron",
        "evidence_class": "local_regression",
        "production_speed_claim": False,
        "hardware_measurement_claimed": False,
        "network_behavior_claimed": False,
        "configuration": {"steps": steps, "repeats": repeats, "dt_ms": 0.1},
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
