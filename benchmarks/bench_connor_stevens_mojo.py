#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Controlled Connor-Stevens Mojo-closure benchmark

"""Measure the Connor-Stevens Python, Rust-engine, and Mojo simulation paths.

This benchmark closes the previously non-executable Mojo lane. It records
source hashes, CPU affinity, host load, runtime versions, event parity, and the
measured voltage-trace error over the established 100-macro-step envelope.
Missing backends or unpinned execution fail unless explicitly allowed.
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

from sc_neurocore.neurons.models import connor_stevens
from sc_neurocore.neurons.models.connor_stevens import ConnorStevensNeuron

REPOSITORY = Path(__file__).resolve().parents[1]
N_STEPS = 100
N_REPEATS = 11
CURRENT = 20.0
MOJO_TRACE_ATOL = 2.0e-6
KERNEL = "connor_stevens_mojo_simulate"
BACKENDS = ("python", "rust", "mojo")
SOURCE_PATHS = (
    "benchmarks/bench_connor_stevens_mojo.py",
    "bridge/sc_neurocore_engine/__init__.py",
    "engine/src/neurons/biophysical.rs",
    "src/sc_neurocore/accel/go/services/connor_stevens.go",
    "src/sc_neurocore/accel/julia/neurons/connor_stevens.jl",
    "src/sc_neurocore/accel/mojo/kernels/connor_stevens.mojo",
    "src/sc_neurocore/accel/rust/safety/connor_stevens.rs",
    "src/sc_neurocore/neurons/models/connor_stevens.py",
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
        available = connor_stevens._HAS_RUST
        return available, "" if available else "Rust engine ConnorStevens symbol unavailable"
    available = connor_stevens._ensure_mojo_loaded()
    return available, "" if available else "compiled libconnor_stevens.so unavailable"


def _measure_backend(
    backend: str,
) -> tuple[float, float, npt.NDArray[np.float64], int, tuple[float, ...]]:
    """Warm one backend, then return timings and final numerical state."""
    ConnorStevensNeuron().simulate(20, CURRENT, backend=backend)
    elapsed_ms: list[float] = []
    trace: npt.NDArray[np.float64] = np.empty(0, dtype=np.float64)
    spikes = 0
    final_state: tuple[float, ...] = ()
    for _repeat in range(N_REPEATS):
        gc.collect()
        neuron = ConnorStevensNeuron()
        started = time.perf_counter_ns()
        trace, spikes = neuron.simulate(N_STEPS, CURRENT, backend=backend)
        elapsed_ms.append((time.perf_counter_ns() - started) / 1_000_000.0)
        final_state = (neuron.v, neuron.m, neuron.h, neuron.n, neuron.a, neuron.b)
    return statistics.median(elapsed_ms), min(elapsed_ms), trace, spikes, final_state


def _runtime_versions() -> dict[str, str]:
    """Record the runtimes involved in the measured closure."""
    home = Path.home()
    mojo = _tool_path("mojo", home / ".pixi/bin/mojo") or ""
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "rust": _tool_version([_tool_path("rustc") or "", "--version"]),
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
    parser = argparse.ArgumentParser(description="Controlled Connor-Stevens Mojo benchmark")
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
            rows[backend] = {
                "available": False,
                "used": False,
                "unavailable_reason": reason,
            }
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
            parity = float(np.max(np.abs(trace - reference)))
        rows[backend] = {
            "available": True,
            "used": True,
            "median_call_ms": median_ms,
            "minimum_call_ms": minimum_ms,
            "speedup_vs_python": (reference_ms / median_ms) if reference_ms is not None else 1.0,
            "parity_max_abs_diff": parity,
            "event_count": spikes,
            "event_count_matches_python": (
                True if reference_spikes is None else spikes == reference_spikes
            ),
            "final_state": dict(zip(("v", "m", "h", "n", "a", "b"), final_state, strict=True)),
        }

    measured_order = sorted(
        (backend for backend in BACKENDS if rows[backend].get("used") is True),
        key=lambda backend: float(rows[backend]["median_call_ms"]),
    )
    report: dict[str, Any] = {
        "schema_version": "sc-neurocore.polyglot-benchmark.v1",
        "kernel": KERNEL,
        "workload": {
            "n_steps": N_STEPS,
            "repeats": N_REPEATS,
            "current": CURRENT,
            "parameters": "Connor-Stevens defaults; 100 candidate-first RK4 sub-steps per macro-step",
            "mojo_trace_atol": MOJO_TRACE_ATOL,
        },
        "meta": _environment(load_start),
        "backends": rows,
        "measured_order": measured_order,
        "source_hashes": _source_hashes(),
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Connor-Stevens benchmark: {N_STEPS} macro steps x {N_REPEATS} repeats")
    for backend in measured_order:
        row = rows[backend]
        print(
            f"{backend:>7}: {float(row['median_call_ms']):10.3f} ms  "
            f"{float(row['speedup_vs_python']):8.2f}x  "
            f"max|delta|={float(row['parity_max_abs_diff']):.3e}  "
            f"events={int(row['event_count'])}"
        )
    print(f"Measured order: {', '.join(measured_order)}")
    print(f"Wrote {args.json}")

    if any(not bool(row.get("event_count_matches_python", True)) for row in rows.values()):
        return 3
    mojo_gap = float(rows.get("mojo", {}).get("parity_max_abs_diff", 0.0))
    return 4 if mojo_gap > MOJO_TRACE_ATOL else 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))
