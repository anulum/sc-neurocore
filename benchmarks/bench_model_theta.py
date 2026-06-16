#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Theta exact-flow local regression benchmark

from __future__ import annotations

import json
from pathlib import Path
import platform
import re
import statistics
import subprocess
import time
from datetime import UTC, datetime
from typing import Any, Protocol

from sc_neurocore.neurons.models.theta import ThetaNeuron


STEPS = 200_000
REPEATS = 5
CURRENT = 0.5
OUTPUT = Path("benchmarks/results/local_python_2026-06-16_theta_exact_flow.json")
GO_BENCH_RE = re.compile(r"^BenchmarkThetaExactFlow-\d+\s+\d+\s+([0-9.]+)\s+ns/op")


class _StepNeuron(Protocol):
    def step(self, current: float) -> int: ...


def _python_neuron() -> _StepNeuron:
    return ThetaNeuron()


def _run_once(factory: Any, backend: str) -> dict[str, object]:
    neuron = factory()
    spikes = 0
    start_ns = time.perf_counter_ns()
    for _ in range(STEPS):
        spikes += int(neuron.step(CURRENT))
    elapsed_ns = time.perf_counter_ns() - start_ns
    return {
        "backend": backend,
        "steps": STEPS,
        "current": CURRENT,
        "elapsed_ns": elapsed_ns,
        "ns_per_step": elapsed_ns / STEPS,
        "spikes": spikes,
    }


def _run_backend(name: str, factory: Any) -> dict[str, object]:
    results = [_run_once(factory, name) for _ in range(REPEATS)]
    ns_per_step = [float(result["ns_per_step"]) for result in results]
    return {
        "backend": name,
        "median_ns_per_step": statistics.median(ns_per_step),
        "min_ns_per_step": min(ns_per_step),
        "max_ns_per_step": max(ns_per_step),
        "results": results,
    }


def _run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, text=True, capture_output=True)


def _run_rust_backend() -> dict[str, object]:
    command = [
        "cargo",
        "run",
        "--manifest-path",
        "engine/Cargo.toml",
        "--example",
        "bench_theta_exact_flow",
    ]
    try:
        completed = _run_command(command)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        return {
            "backend": "rust",
            "skipped": True,
            "reason": f"Rust theta benchmark command failed: {exc}",
        }
    report = json.loads(completed.stdout)
    report["driver_command"] = " ".join(command)
    return report


def _run_go_backend() -> dict[str, object]:
    command = [
        "go",
        "test",
        "src/sc_neurocore/accel/go/services/theta.go",
        "src/sc_neurocore/accel/go/services/theta_test.go",
        "-run",
        "^$",
        "-bench",
        "BenchmarkThetaExactFlow$",
        "-benchtime",
        "200000x",
        "-count",
        str(REPEATS),
    ]
    try:
        completed = _run_command(command)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        return {
            "backend": "go",
            "skipped": True,
            "reason": f"Go theta benchmark command failed: {exc}",
        }
    values = [
        float(match.group(1))
        for line in completed.stdout.splitlines()
        if (match := GO_BENCH_RE.match(line))
    ]
    if not values:
        return {
            "backend": "go",
            "skipped": True,
            "reason": "Go benchmark output did not contain BenchmarkThetaExactFlow ns/op rows",
            "stdout": completed.stdout,
        }
    return {
        "backend": "go",
        "command": " ".join(command),
        "steps": STEPS,
        "repeats": len(values),
        "current": CURRENT,
        "median_ns_per_step": statistics.median(values),
        "min_ns_per_step": min(values),
        "max_ns_per_step": max(values),
        "results_ns_per_step": values,
    }


def _run_julia_backend() -> dict[str, object]:
    script = f"""
using Statistics
include("src/sc_neurocore/accel/julia/neurons/theta.jl")
const STEPS = {STEPS}
const REPEATS = {REPEATS}
const CURRENT = {CURRENT}
function run_once()
    s = ThetaAccel.ThetaNeuronState()
    spikes = 0
    start = time_ns()
    for _ in 1:STEPS
        spikes += ThetaAccel.step!(s, CURRENT)
    end
    elapsed = time_ns() - start
    return elapsed / STEPS, spikes, s.theta
end
results = [run_once() for _ in 1:REPEATS]
values = [r[1] for r in results]
println("median_ns_per_step=", median(values))
println("min_ns_per_step=", minimum(values))
println("max_ns_per_step=", maximum(values))
println("results_ns_per_step=", join(values, ","))
println("spike_counts=", join([r[2] for r in results], ","))
println("final_thetas=", join([r[3] for r in results], ","))
"""
    command = ["julia", "--project=.", "-e", script]
    try:
        completed = _run_command(command)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        return {
            "backend": "julia",
            "skipped": True,
            "reason": f"Julia theta benchmark command failed: {exc}",
        }
    fields = dict(line.split("=", 1) for line in completed.stdout.splitlines() if "=" in line)
    values = [float(value) for value in fields["results_ns_per_step"].split(",")]
    return {
        "backend": "julia",
        "command": "julia --project=. -e <theta exact-flow benchmark>",
        "steps": STEPS,
        "repeats": REPEATS,
        "current": CURRENT,
        "median_ns_per_step": float(fields["median_ns_per_step"]),
        "min_ns_per_step": float(fields["min_ns_per_step"]),
        "max_ns_per_step": float(fields["max_ns_per_step"]),
        "results_ns_per_step": values,
        "spike_counts": [int(value) for value in fields["spike_counts"].split(",")],
        "final_thetas": [float(value) for value in fields["final_thetas"].split(",")],
    }


def main() -> int:
    report = {
        "spdx_license": "AGPL-3.0-or-later",
        "commercial_license": "available",
        "copyright_concepts": "© Concepts 1996–2026 Miroslav Šotek. All rights reserved.",
        "copyright_code": "© Code 2020–2026 Miroslav Šotek. All rights reserved.",
        "orcid": "0009-0009-3560-0851",
        "contact": "www.anulum.li | protoscience@anulum.li",
        "benchmark": "ThetaNeuron tangent-half-angle exact constant-current flow step",
        "timestamp_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "command": "PYTHONPATH=src ./.venv/bin/python benchmarks/bench_model_theta.py",
        "evidence_class": "local_regression_non_isolated",
        "production_speed_claim": False,
        "isolation": "non-isolated loaded workstation",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "steps": STEPS,
        "repeats": REPEATS,
        "current": CURRENT,
        "results": [
            _run_backend("python", _python_neuron),
            _run_rust_backend(),
            _run_go_backend(),
            _run_julia_backend(),
            {
                "backend": "mojo",
                "skipped": True,
                "reason": "Mojo theta surface is a spike-kernel mirror without a stateful benchmark hook",
            },
        ],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
