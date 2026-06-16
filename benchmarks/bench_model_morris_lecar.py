#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Morris-Lecar RK4 multi-backend local regression benchmark

from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import platform
import re
import statistics
import subprocess
import time
from typing import Any, Protocol, cast

from sc_neurocore.neurons.models.morris_lecar import MorrisLecarNeuron


STEPS = 200_000
REPEATS = 5
CURRENT = 100.0
OUTPUT = Path("benchmarks/results/local_python_2026-06-17_morris_lecar_rk4.json")
REPO_ROOT = Path(__file__).resolve().parents[1]
GO_BENCH_RE = re.compile(r"^BenchmarkMorrisLecarRK4-\d+\s+\d+\s+([0-9.]+)\s+ns/op")
SOURCE_HASH_PATHS = {
    "benchmarks/bench_model_morris_lecar.py": REPO_ROOT / "benchmarks/bench_model_morris_lecar.py",
    "engine/Cargo.toml": REPO_ROOT / "engine/Cargo.toml",
    "engine/examples/bench_morris_lecar_rk4.rs": REPO_ROOT
    / "engine/examples/bench_morris_lecar_rk4.rs",
    "engine/src/neurons/simple_spiking.rs": REPO_ROOT / "engine/src/neurons/simple_spiking.rs",
    "src/sc_neurocore/neurons/models/morris_lecar.py": REPO_ROOT
    / "src/sc_neurocore/neurons/models/morris_lecar.py",
    "src/sc_neurocore/accel/go/services/morris_lecar.go": REPO_ROOT
    / "src/sc_neurocore/accel/go/services/morris_lecar.go",
    "src/sc_neurocore/accel/go/services/morris_lecar_test.go": REPO_ROOT
    / "src/sc_neurocore/accel/go/services/morris_lecar_test.go",
    "src/sc_neurocore/accel/julia/neurons/morris_lecar.jl": REPO_ROOT
    / "src/sc_neurocore/accel/julia/neurons/morris_lecar.jl",
    "src/sc_neurocore/accel/rust/safety/morris_lecar.rs": REPO_ROOT
    / "src/sc_neurocore/accel/rust/safety/morris_lecar.rs",
}


class _StepNeuron(Protocol):
    v: float
    w: float

    def step(self, current: float) -> int: ...


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_hashes() -> dict[str, object]:
    flat = {source: _sha256(path) for source, path in SOURCE_HASH_PATHS.items()}
    nested: dict[str, object] = dict(flat)
    for source, path in SOURCE_HASH_PATHS.items():
        stem, extension = source.rsplit(".", 1)
        existing = nested.get(stem)
        if isinstance(existing, dict):
            existing[extension] = _sha256(path)
        else:
            nested[stem] = {extension: _sha256(path)}
    return nested


def _run_once(factory: Any, backend: str) -> dict[str, object]:
    neuron: _StepNeuron = factory()
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
        "ending_state": [float(neuron.v), float(neuron.w)],
    }


def _run_python_backend() -> dict[str, object]:
    results = [_run_once(lambda: MorrisLecarNeuron(), "python") for _ in range(REPEATS)]
    ns_per_step = [cast(float, result["ns_per_step"]) for result in results]
    first_spikes = cast(int, results[0]["spikes"])
    return {
        "backend": "python",
        "median_ns_per_step": statistics.median(ns_per_step),
        "min_ns_per_step": min(ns_per_step),
        "max_ns_per_step": max(ns_per_step),
        "spikes": first_spikes,
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
        "bench_morris_lecar_rk4",
    ]
    try:
        completed = _run_command(command)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        return {"backend": "rust", "skipped": True, "reason": f"Rust benchmark failed: {exc}"}
    report = cast(dict[str, object], json.loads(completed.stdout))
    report["driver_command"] = " ".join(command)
    return report


def _run_go_backend() -> dict[str, object]:
    command = [
        "go",
        "test",
        "src/sc_neurocore/accel/go/services/morris_lecar.go",
        "src/sc_neurocore/accel/go/services/morris_lecar_test.go",
        "-run",
        "^$",
        "-bench",
        "BenchmarkMorrisLecarRK4$",
        "-benchtime",
        "200000x",
        "-count",
        str(REPEATS),
    ]
    try:
        completed = _run_command(command)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        return {"backend": "go", "skipped": True, "reason": f"Go benchmark failed: {exc}"}
    values = [
        float(match.group(1))
        for line in completed.stdout.splitlines()
        if (match := GO_BENCH_RE.match(line))
    ]
    if not values:
        return {
            "backend": "go",
            "skipped": True,
            "reason": "Go benchmark output did not include BenchmarkMorrisLecarRK4 ns/op rows",
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
        "spikes": 0,
    }


def _run_julia_backend() -> dict[str, object]:
    script = f"""
using Statistics
include("src/sc_neurocore/accel/julia/neurons/morris_lecar.jl")
const STEPS = {STEPS}
const REPEATS = {REPEATS}
const CURRENT = {CURRENT}
function run_once()
    s = MorrisLecarAccel.MorrisLecarNeuronState()
    spikes = 0
    start = time_ns()
    for _ in 1:STEPS
        spikes += MorrisLecarAccel.step!(s, CURRENT)
    end
    elapsed = time_ns() - start
    return elapsed / STEPS, spikes, s.v, s.w
end
results = [run_once() for _ in 1:REPEATS]
values = [r[1] for r in results]
println("median_ns_per_step=", median(values))
println("min_ns_per_step=", minimum(values))
println("max_ns_per_step=", maximum(values))
println("results_ns_per_step=", join(values, ","))
println("spike_counts=", join([r[2] for r in results], ","))
println("final_vs=", join([r[3] for r in results], ","))
println("final_ws=", join([r[4] for r in results], ","))
"""
    command = ["julia", "--project=.", "-e", script]
    try:
        completed = _run_command(command)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        return {"backend": "julia", "skipped": True, "reason": f"Julia benchmark failed: {exc}"}
    fields = dict(line.split("=", 1) for line in completed.stdout.splitlines() if "=" in line)
    values = [float(value) for value in fields["results_ns_per_step"].split(",")]
    return {
        "backend": "julia",
        "command": "julia --project=. -e <morris-lecar rk4 benchmark>",
        "steps": STEPS,
        "repeats": len(values),
        "current": CURRENT,
        "median_ns_per_step": float(fields["median_ns_per_step"]),
        "min_ns_per_step": float(fields["min_ns_per_step"]),
        "max_ns_per_step": float(fields["max_ns_per_step"]),
        "results_ns_per_step": values,
        "spikes": int(fields["spike_counts"].split(",")[0]),
        "final_vs": [float(value) for value in fields["final_vs"].split(",")],
        "final_ws": [float(value) for value in fields["final_ws"].split(",")],
    }


def main() -> None:
    python = _run_python_backend()
    rust = _run_rust_backend()
    go = _run_go_backend()
    julia = _run_julia_backend()
    payload = {
        "spdx_license": "AGPL-3.0-or-later",
        "benchmark": "MorrisLecarNeuron candidate-first RK4 conductance step",
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "command": "PYTHONPATH=src .venv/bin/python benchmarks/bench_model_morris_lecar.py",
        "evidence_class": "local_regression_non_isolated",
        "production_speed_claim": False,
        "hardware_measurement_claimed": False,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "steps": STEPS,
        "repeats": REPEATS,
        "current": CURRENT,
        "backend_summary": {
            "python": python,
            "rust": rust,
            "go": go,
            "julia": julia,
            "mojo": {
                "backend": "mojo",
                "skipped": True,
                "reason": "No maintained Morris-Lecar Mojo runtime counterpart exists in this checkout.",
            },
        },
        "source_hashes": _source_hashes(),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
