#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Controlled Quadratic IF five-backend benchmark

"""Measure the maintained Quadratic IF exact flow through public dispatch.

The evidence record binds public-API timings to source hashes, CPU affinity,
host load, runtime versions, exact event parity, bounded trace parity, final
state, and an executable Rust-safety verification. Missing backends or
unpinned execution fail unless explicitly acknowledged.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import shutil
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from sc_neurocore.accel import quadratic_if as backends
from sc_neurocore.neurons.models.quadratic_if import QuadraticIFNeuron

REPOSITORY = Path(__file__).resolve().parents[1]
N_STEPS = 100_000
N_REPEATS = 7
CURRENT = 5.0
TRACE_ATOL = 2.0e-12
KERNEL = "quadratic_if_exact_constant_current_flow"
BACKENDS = ("python", "rust", "julia", "go", "mojo")
SOURCE_PATHS = (
    "benchmarks/bench_model_quadratic_if.py",
    "engine/src/neurons/trivial.rs",
    "src/sc_neurocore/accel/go/neurons/quadratic_if/quadratic_if.go",
    "src/sc_neurocore/accel/go/services/quadratic_if.go",
    "src/sc_neurocore/accel/go/services/quadratic_if_test.go",
    "src/sc_neurocore/accel/julia/neurons/quadratic_if.jl",
    "src/sc_neurocore/accel/julia/quadratic_if_parity_test.jl",
    "src/sc_neurocore/accel/mojo/kernels/quadratic_if.mojo",
    "src/sc_neurocore/accel/rust/examples/quadratic_if_trace.rs",
    "src/sc_neurocore/accel/rust/safety/quadratic_if.rs",
    "src/sc_neurocore/neurons/models/quadratic_if.py",
    "src/sc_neurocore/accel/quadratic_if.py",
)


def _cpu_model() -> str:
    """Return the first Linux CPU model string, or a portable fallback."""
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text(encoding="utf-8")
    except OSError:
        cpuinfo = ""
    for line in cpuinfo.splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def _read_optional(path: Path) -> str:
    """Read one host metadata file without making it a dependency."""
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return "unavailable"


def _tool_path(name: str, fallback: Path | None = None) -> str | None:
    """Resolve a runtime executable with one explicit fallback."""
    resolved = shutil.which(name)
    if resolved is not None:
        return resolved
    if fallback is not None and fallback.is_file():
        return str(fallback)
    return None


def _tool_version(command: list[str]) -> str:
    """Return the first version line for a runtime executable."""
    if not command or command[0] == "":
        return "unavailable"
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unavailable"
    output = result.stdout.strip() or result.stderr.strip()
    return output.splitlines()[0] if output else f"exit {result.returncode}"


def _source_hashes() -> dict[str, object]:
    """Hash every implementation and ABI surface relevant to this closure."""
    flat = {
        relative: hashlib.sha256((REPOSITORY / relative).read_bytes()).hexdigest()
        for relative in SOURCE_PATHS
    }
    nested: dict[str, object] = {}
    for relative, digest in flat.items():
        stem, suffix = relative.rsplit(".", 1)
        nested[stem] = {suffix: digest}
    return {**flat, **nested}


def _probe_backend(backend: str) -> tuple[bool, str]:
    """Return backend availability plus a deterministic diagnostic."""
    if backend == "python":
        return True, ""
    if backend == "rust":
        return (
            backends._HAS_RUST,
            "" if backends._HAS_RUST else "Rust engine QuadraticIF symbol unavailable",
        )
    if backend == "julia":
        available = backends.ensure_julia_loaded()
        return available, "" if available else "juliacall or Quadratic IF module unavailable"
    if backend == "go":
        available = backends.ensure_go_loaded()
        return available, "" if available else "compiled Go libquadratic_if.so unavailable"
    available = backends.ensure_mojo_loaded()
    return available, "" if available else "compiled Mojo libquadratic_if.so unavailable"


def _measure_backend(
    backend: str,
) -> tuple[float, float, npt.NDArray[np.float64], int, tuple[float]]:
    """Warm one backend, then return timings and final numerical state."""
    QuadraticIFNeuron().simulate(20, CURRENT, backend=backend)
    elapsed_ms: list[float] = []
    trace: npt.NDArray[np.float64] = np.empty(0, dtype=np.float64)
    spikes = 0
    final_state = (-1.0,)
    for _repeat in range(N_REPEATS):
        gc.collect()
        neuron = QuadraticIFNeuron()
        started = time.perf_counter_ns()
        trace, spikes = neuron.simulate(N_STEPS, CURRENT, backend=backend)
        elapsed_ms.append((time.perf_counter_ns() - started) / 1_000_000.0)
        final_state = (neuron.v,)
    return statistics.median(elapsed_ms), min(elapsed_ms), trace, spikes, final_state


def _verify_rust_safety() -> dict[str, Any]:
    """Execute the actual accel/rust/safety QIF test module."""
    command = [
        "cargo",
        "test",
        "--release",
        "--manifest-path",
        "src/sc_neurocore/accel/rust/Cargo.toml",
        "quadratic_if::tests",
    ]
    completed = subprocess.run(
        command,
        cwd=REPOSITORY,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    output = (completed.stdout + "\n" + completed.stderr).strip().splitlines()
    return {
        "command": " ".join(command),
        "passed": completed.returncode == 0,
        "returncode": completed.returncode,
        "output_tail": output[-12:],
    }


def _runtime_versions() -> dict[str, str]:
    """Record all runtimes involved in the measured closure."""
    home = Path.home()
    mojo = _tool_path("mojo", home / ".pixi/bin/mojo") or ""
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "rust": _tool_version([_tool_path("rustc") or "", "--version"]),
        "julia": _tool_version([_tool_path("julia") or "", "--version"]),
        "go": _tool_version([_tool_path("go") or "", "version"]),
        "mojo": _tool_version([mojo, "--version"]),
    }


def _environment(load_start: tuple[float, float, float]) -> dict[str, Any]:
    """Capture affinity and load without claiming kernel isolation."""
    affinity = sorted(os.sched_getaffinity(0))
    cpu = affinity[0] if len(affinity) == 1 else None
    governor = (
        _read_optional(Path(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor"))
        if cpu is not None
        else "mixed-or-unpinned"
    )
    return {
        "cpu": _cpu_model(),
        "platform": platform.platform(),
        "affinity": affinity,
        "single_cpu_pinned": len(affinity) == 1,
        "kernel_isolated_cpus": _read_optional(Path("/sys/devices/system/cpu/isolated")),
        "governor": governor,
        "load_average_start": list(load_start),
        "load_average_end": list(os.getloadavg()),
        "measurement_scope": (
            "single-logical-CPU affinity; kernel isolation and workstation load reported separately"
        ),
        "runtime_versions": _runtime_versions(),
    }


def main(argv: list[str]) -> int:
    """Run the controlled benchmark and write its evidence artefact."""
    parser = argparse.ArgumentParser(description="Controlled Quadratic IF five-backend benchmark")
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--allow-unpinned", action="store_true")
    parser.add_argument("--allow-unavailable-backends", action="store_true")
    args = parser.parse_args(argv)

    affinity = sorted(os.sched_getaffinity(0))
    if len(affinity) != 1 and not args.allow_unpinned:
        print(f"Refusing unpinned benchmark; affinity is {affinity}")
        return 2

    load_start = os.getloadavg()
    probes = {backend: _probe_backend(backend) for backend in BACKENDS}
    missing = [backend for backend, (available, _reason) in probes.items() if not available]
    if missing and not args.allow_unavailable_backends:
        print("Missing required backend(s): " + ", ".join(missing))
        return 2

    rows: dict[str, dict[str, Any]] = {}
    reference: npt.NDArray[np.float64] | None = None
    reference_ms: float | None = None
    reference_spikes: int | None = None
    for backend in BACKENDS:
        available, reason = probes[backend]
        if not available:
            rows[backend] = {"available": False, "used": False, "unavailable_reason": reason}
            continue
        median_ms, minimum_ms, trace, spikes, final_state = _measure_backend(backend)
        if backend == "python":
            reference = trace
            reference_ms = median_ms
            reference_spikes = spikes
            parity = 0.0
        else:
            if reference is None or reference_ms is None or reference_spikes is None:
                raise RuntimeError("Python reference must be measured first")
            parity = float(np.max(np.abs(trace - reference))) if trace.size else 0.0
        rows[backend] = {
            "available": True,
            "used": True,
            "median_call_ms": median_ms,
            "minimum_call_ms": minimum_ms,
            "speedup_vs_python": reference_ms / median_ms if reference_ms is not None else 1.0,
            "parity_max_abs_diff": parity,
            "event_count": spikes,
            "event_count_matches_python": (
                True if reference_spikes is None else spikes == reference_spikes
            ),
            "final_state": dict(zip(("v",), final_state, strict=True)),
        }

    measured_order = sorted(
        (backend for backend in BACKENDS if rows[backend].get("used") is True),
        key=lambda backend: float(rows[backend]["median_call_ms"]),
    )
    rust_safety = _verify_rust_safety()
    report: dict[str, Any] = {
        "schema_version": "sc-neurocore.polyglot-benchmark.v1",
        "kernel": KERNEL,
        "workload": {
            "n_steps": N_STEPS,
            "repeats": N_REPEATS,
            "current": CURRENT,
            "parameters": "Quadratic IF factory defaults; exact constant-current Riccati flow",
            "trace_atol": TRACE_ATOL,
        },
        "meta": _environment(load_start),
        "backends": rows,
        "measured_order": measured_order,
        "verification": {"rust_safety": rust_safety},
        "source_hashes": _source_hashes(),
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Quadratic IF benchmark: {N_STEPS} steps x {N_REPEATS} repeats")
    for backend in measured_order:
        row = rows[backend]
        print(
            f"{backend:>7}: {float(row['median_call_ms']):10.3f} ms  "
            f"{float(row['speedup_vs_python']):8.2f}x  "
            f"max|delta|={float(row['parity_max_abs_diff']):.3e}  "
            f"events={int(row['event_count'])}"
        )
    print(f"Measured order: {', '.join(measured_order)}")
    print(f"Rust safety tests: {'PASS' if rust_safety['passed'] else 'FAIL'}")
    print(f"Wrote {args.json}")

    if not rust_safety["passed"]:
        return 5
    if any(not bool(row.get("event_count_matches_python", True)) for row in rows.values()):
        return 3
    if any(
        float(row.get("parity_max_abs_diff", 0.0)) > TRACE_ATOL
        for row in rows.values()
        if row.get("used") is True
    ):
        return 4
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))
