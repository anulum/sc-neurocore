#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Controlled Hodgkin-Huxley Mojo-closure benchmark

"""Measure all maintained Hodgkin-Huxley default-runtime paths.

The record binds timings to source hashes, CPU affinity, host load, runtime
versions, event parity, complete final state, and voltage-trace error over the
enrolled 100-macro-step baseline-Euler envelope. Missing backends or unpinned
execution fail unless the operator explicitly permits them.
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
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from sc_neurocore.neurons.models import hodgkin_huxley
from sc_neurocore.neurons.models.hodgkin_huxley import HodgkinHuxleyNeuron

REPOSITORY = Path(__file__).resolve().parents[1]
N_STEPS = 100
N_REPEATS = 11
CURRENT = 20.0
MOJO_TRACE_ATOL = 2.0e-9
KERNEL = "hodgkin_huxley_five_runtime"
BACKENDS = ("python", "rust", "julia", "go", "mojo")
PARITY_ATOL = {
    "python": 0.0,
    "rust": 1.0e-9,
    "julia": 1.0e-9,
    "go": 1.0e-9,
    "mojo": MOJO_TRACE_ATOL,
}
SOURCE_PATHS = (
    "benchmarks/bench_hodgkin_huxley_mojo.py",
    "bridge/sc_neurocore_engine/__init__.py",
    "engine/src/bindings/biophysical/hodgkin_huxley.rs",
    "engine/src/neurons/biophysical/hodgkin_huxley.rs",
    "src/sc_neurocore/accel/go/services/hodgkin_huxley_test.go",
    "src/sc_neurocore/accel/go/services/hodgkin_huxley.go",
    "src/sc_neurocore/accel/julia/hodgkin_huxley_parity_test.jl",
    "src/sc_neurocore/accel/julia/neurons/hodgkin_huxley.jl",
    "src/sc_neurocore/accel/mojo/kernels/hodgkin_huxley.mojo",
    "src/sc_neurocore/accel/rust/safety/hodgkin_huxley.rs",
    "src/sc_neurocore/neurons/model_descriptors/HodgkinHuxleyNeuron.toml",
    "src/sc_neurocore/neurons/model_schemas/hodgkin_huxley.json",
    "src/sc_neurocore/neurons/model_schemas/hodgkin_huxley.toml",
    "src/sc_neurocore/neurons/models/hodgkin_huxley.py",
    "src/sc_neurocore/neurons/reference_receipts/hodgkin_huxley_1952.json",
    "src/sc_neurocore/neurons/reference_trace_data/hodgkin_huxley_driven_spiking_doi.json",
)

GO_ROOT = REPOSITORY / "src/sc_neurocore/accel/go"
JULIA_SOURCE = REPOSITORY / "src/sc_neurocore/accel/julia/neurons/hodgkin_huxley.jl"

_GO_HELPER = r"""package main

import (
    "encoding/json"
    "os"
    "strconv"
    "time"

    services "github.com/anulum/sc-neurocore/accel/services"
)

type report struct {
    ResultsMS []float64
    Trace []float64
    EventCount int
    FinalState []float64
}

func run(nSteps int, current float64) ([]float64, int, []float64) {
    neuron := services.NewHodgkinHuxleyNeuron()
    trace := make([]float64, nSteps)
    events := 0
    for index := 0; index < nSteps; index++ {
        event, err := neuron.Step(current)
        if err != nil { panic(err) }
        events += event
        trace[index] = neuron.V
    }
    state := []float64{neuron.V, neuron.M, neuron.H, neuron.N}
    return trace, events, state
}

func main() {
    nSteps, _ := strconv.Atoi(os.Args[1])
    current, _ := strconv.ParseFloat(os.Args[2], 64)
    repeats, _ := strconv.Atoi(os.Args[3])
    run(nSteps, current)
    result := report{}
    for index := 0; index < repeats; index++ {
        started := time.Now()
        result.Trace, result.EventCount, result.FinalState = run(nSteps, current)
        result.ResultsMS = append(result.ResultsMS, float64(time.Since(started).Nanoseconds())/1e6)
    }
    if err := json.NewEncoder(os.Stdout).Encode(result); err != nil { panic(err) }
}
"""

_JULIA_HELPER = r"""
include(ARGS[1])
using .HodgkinHuxleyAccel

function run_once(n_steps::Int, current::Float64)
    neuron = HodgkinHuxleyNeuronState()
    trace = zeros(Float64, n_steps)
    events = 0
    for index in 1:n_steps
        event = step!(neuron, current)
        event < 0 && error("Hodgkin-Huxley Julia kernel rejected enrolled input")
        events += event
        trace[index] = neuron.v
    end
    state = (neuron.v, neuron.m, neuron.h, neuron.n)
    return trace, events, state
end

function main()
    n_steps = parse(Int, ARGS[2])
    current = parse(Float64, ARGS[3])
    repeats = parse(Int, ARGS[4])
    run_once(n_steps, current)
    times = Float64[]
    trace = Float64[]
    events = 0
    state = ntuple(_ -> 0.0, 4)
    for _ in 1:repeats
        started = time_ns()
        trace, events, state = run_once(n_steps, current)
        push!(times, (time_ns() - started) / 1.0e6)
    end
    println(join(times, ','))
    println(join(trace, ','))
    println(events)
    println(join(state, ','))
end

main()
"""


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
        bucket = nested.setdefault(stem, {})
        if not isinstance(bucket, dict):
            raise RuntimeError(f"source-hash namespace collision at {stem}")
        bucket[suffix] = digest
    return {**flat, **nested}


def _probe_backend(backend: str) -> tuple[bool, str]:
    """Return backend availability plus a deterministic diagnostic."""
    if backend == "python":
        return True, ""
    if backend == "rust":
        available = hodgkin_huxley._HAS_RUST
        return available, "" if available else "Rust engine HodgkinHuxley symbol unavailable"
    if backend == "julia":
        available = shutil.which("julia") is not None and JULIA_SOURCE.is_file()
        return available, "" if available else "Julia runtime or Hodgkin-Huxley module unavailable"
    if backend == "go":
        available = shutil.which("go") is not None and GO_ROOT.is_dir()
        return available, "" if available else "Go runtime or accelerator module unavailable"
    available = hodgkin_huxley._ensure_mojo_loaded()
    return available, "" if available else "compiled libhodgkin_huxley.so unavailable"


def _measure_in_process(
    backend: str,
) -> tuple[float, float, npt.NDArray[np.float64], int, tuple[float, ...]]:
    """Warm one backend, then return timings and final numerical state."""
    HodgkinHuxleyNeuron().simulate(20, CURRENT, backend=backend)
    elapsed_ms: list[float] = []
    trace: npt.NDArray[np.float64] = np.empty(0, dtype=np.float64)
    spikes = 0
    final_state: tuple[float, ...] = ()
    for _repeat in range(N_REPEATS):
        gc.collect()
        neuron = HodgkinHuxleyNeuron()
        started = time.perf_counter_ns()
        trace, spikes = neuron.simulate(N_STEPS, CURRENT, backend=backend)
        elapsed_ms.append((time.perf_counter_ns() - started) / 1_000_000.0)
        final_state = (neuron.v, neuron.m, neuron.h, neuron.n)
    return statistics.median(elapsed_ms), min(elapsed_ms), trace, spikes, final_state


def _measure_go() -> tuple[float, float, npt.NDArray[np.float64], int, tuple[float, ...]]:
    """Execute the maintained Go service and return its measured complete state."""
    with tempfile.TemporaryDirectory(prefix="scn-hodgkin-go-") as tmpdir:
        helper = Path(tmpdir) / "main.go"
        helper.write_text(_GO_HELPER, encoding="utf-8")
        completed = subprocess.run(
            ["go", "run", str(helper), str(N_STEPS), repr(CURRENT), str(N_REPEATS)],
            cwd=GO_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=300,
        )
    payload = json.loads(completed.stdout)
    timings = [float(value) for value in payload["ResultsMS"]]
    return (
        statistics.median(timings),
        min(timings),
        np.asarray(payload["Trace"], dtype=np.float64),
        int(payload["EventCount"]),
        tuple(float(value) for value in payload["FinalState"]),
    )


def _measure_julia() -> tuple[float, float, npt.NDArray[np.float64], int, tuple[float, ...]]:
    """Execute the maintained Julia module and return its measured complete state."""
    completed = subprocess.run(
        [
            "julia",
            "--startup-file=no",
            "-e",
            _JULIA_HELPER,
            str(JULIA_SOURCE),
            str(N_STEPS),
            repr(CURRENT),
            str(N_REPEATS),
        ],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    )
    lines = completed.stdout.strip().splitlines()
    if len(lines) != 4:
        raise RuntimeError(f"unexpected Julia Hodgkin-Huxley output: {completed.stdout!r}")
    timings = [float(value) for value in lines[0].split(",")]
    trace = np.asarray([float(value) for value in lines[1].split(",")], dtype=np.float64)
    final_state = tuple(float(value) for value in lines[3].split(","))
    return statistics.median(timings), min(timings), trace, int(lines[2]), final_state


def _measure_backend(
    backend: str,
) -> tuple[float, float, npt.NDArray[np.float64], int, tuple[float, ...]]:
    """Measure one maintained runtime through its executable surface."""
    if backend == "go":
        return _measure_go()
    if backend == "julia":
        return _measure_julia()
    return _measure_in_process(backend)


def _runtime_versions() -> dict[str, str]:
    """Record the runtimes involved in the measured closure."""
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
    parser = argparse.ArgumentParser(description="Controlled Hodgkin-Huxley Mojo benchmark")
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
    reference_state: tuple[float, ...] | None = None
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
            reference_state = final_state
            parity = 0.0
        else:
            if (
                reference is None
                or reference_ms is None
                or reference_spikes is None
                or reference_state is None
            ):
                raise RuntimeError("Python reference must be measured first")
            parity = float(np.max(np.abs(trace - reference)))
        if reference_state is None:
            raise RuntimeError("Python reference state must be measured first")
        final_state_gap = max(
            abs(actual - expected)
            for actual, expected in zip(final_state, reference_state, strict=True)
        )
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
            "final_state_max_abs_diff": final_state_gap,
            "final_state_matches_python": final_state_gap <= PARITY_ATOL[backend],
            "final_state": dict(zip(("v", "m", "h", "n"), final_state, strict=True)),
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
            "parameters": (
                "Hodgkin-Huxley defaults; 100 gate-first baseline-Euler substeps per macro-step"
            ),
            "mojo_trace_atol": MOJO_TRACE_ATOL,
        },
        "meta": _environment(load_start),
        "backends": rows,
        "measured_order": measured_order,
        "source_hashes": _source_hashes(),
        "evidence_class": "local_regression_non_isolated",
        "production_speed_claim": False,
        "hardware_measurement_claimed": False,
        "notes": "Loaded-host regression only; timings are not comparative production claims.",
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Hodgkin-Huxley benchmark: {N_STEPS} macro steps x {N_REPEATS} repeats")
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
    parity_failed = any(
        bool(row.get("used"))
        and (
            float(row.get("parity_max_abs_diff", 0.0)) > PARITY_ATOL[backend]
            or not bool(row.get("final_state_matches_python", False))
        )
        for backend, row in rows.items()
    )
    return 4 if parity_failed else 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))
