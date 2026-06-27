#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Hay L5 RK4 multi-backend local regression benchmark

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
from typing import Protocol, cast

from sc_neurocore.neurons.models.hay_l5 import HayL5PyramidalNeuron

STEPS = 20_000
REPEATS = 5
CURRENT_SOMA = 10.0
CURRENT_TUFT = 0.0
OUTPUT = Path("benchmarks/results/local_python_2026-06-26_hay_l5_rk4.json")
REPO_ROOT = Path(__file__).resolve().parents[1]
GO_BENCH_RE = re.compile(r"^BenchmarkHayL5RK4-\d+\s+\d+\s+([0-9.]+)\s+ns/op")
GO_SPIKES_RE = re.compile(r"\s([0-9.]+)\s+spikes(?:\s|$)")
SOURCE_HASH_PATHS = {
    "benchmarks/bench_model_hay_l5.py": REPO_ROOT / "benchmarks/bench_model_hay_l5.py",
    "engine/Cargo.toml": REPO_ROOT / "engine/Cargo.toml",
    "engine/examples/bench_hay_l5_rk4.rs": REPO_ROOT / "engine/examples/bench_hay_l5_rk4.rs",
    "engine/src/neurons/multi_compartment.rs": REPO_ROOT
    / "engine/src/neurons/multi_compartment.rs",
    "src/sc_neurocore/neurons/models/hay_l5.py": REPO_ROOT
    / "src/sc_neurocore/neurons/models/hay_l5.py",
    "src/sc_neurocore/accel/go/services/hay_l5.go": REPO_ROOT
    / "src/sc_neurocore/accel/go/services/hay_l5.go",
    "src/sc_neurocore/accel/go/services/hay_l5_test.go": REPO_ROOT
    / "src/sc_neurocore/accel/go/services/hay_l5_test.go",
    "src/sc_neurocore/accel/julia/neurons/hay_l5.jl": REPO_ROOT
    / "src/sc_neurocore/accel/julia/neurons/hay_l5.jl",
    "src/sc_neurocore/accel/mojo/kernels/hay_l5.mojo": REPO_ROOT
    / "src/sc_neurocore/accel/mojo/kernels/hay_l5.mojo",
}


class _StepNeuron(Protocol):
    v_s: float
    ca_a: float

    def step(self, current_soma: float, current_tuft: float = 0.0) -> int: ...


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_hashes() -> dict[str, str]:
    return {source: _sha256(path) for source, path in SOURCE_HASH_PATHS.items()}


def _run_once(backend: str) -> dict[str, object]:
    neuron: _StepNeuron = HayL5PyramidalNeuron()
    spikes = 0
    start_ns = time.perf_counter_ns()
    for _ in range(STEPS):
        spikes += int(neuron.step(CURRENT_SOMA, CURRENT_TUFT))
    elapsed_ns = time.perf_counter_ns() - start_ns
    return {
        "backend": backend,
        "steps": STEPS,
        "current_soma": CURRENT_SOMA,
        "current_tuft": CURRENT_TUFT,
        "elapsed_ns": elapsed_ns,
        "ns_per_step": elapsed_ns / STEPS,
        "spikes": spikes,
        "ending_state": [float(neuron.v_s), float(neuron.ca_a)],
    }


def _run_python_backend() -> dict[str, object]:
    results = [_run_once("python") for _ in range(REPEATS)]
    ns_per_step = [cast(float, result["ns_per_step"]) for result in results]
    return {
        "backend": "python",
        "median_ns_per_step": statistics.median(ns_per_step),
        "min_ns_per_step": min(ns_per_step),
        "max_ns_per_step": max(ns_per_step),
        "spikes": cast(int, results[0]["spikes"]),
        "results": results,
    }


def _run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, text=True, capture_output=True)


def _run_rust_backend() -> dict[str, object]:
    command = [
        "cargo",
        "run",
        "--release",
        "--manifest-path",
        "engine/Cargo.toml",
        "--example",
        "bench_hay_l5_rk4",
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
        "src/sc_neurocore/accel/go/services/hay_l5.go",
        "src/sc_neurocore/accel/go/services/hay_l5_test.go",
        "-run",
        "^$",
        "-bench",
        "BenchmarkHayL5RK4$",
        "-benchtime",
        "20000x",
        "-count",
        str(REPEATS),
    ]
    try:
        completed = _run_command(command)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        return {"backend": "go", "skipped": True, "reason": f"Go benchmark failed: {exc}"}
    values: list[float] = []
    spike_counts: list[int] = []
    for line in completed.stdout.splitlines():
        if match := GO_BENCH_RE.match(line):
            values.append(float(match.group(1)))
            if spike_match := GO_SPIKES_RE.search(line):
                spike_counts.append(int(float(spike_match.group(1))))
    if not values:
        return {"backend": "go", "skipped": True, "reason": "Go benchmark produced no rows"}
    return {
        "backend": "go",
        "command": " ".join(command),
        "steps": STEPS,
        "repeats": len(values),
        "current_soma": CURRENT_SOMA,
        "current_tuft": CURRENT_TUFT,
        "median_ns_per_step": statistics.median(values),
        "min_ns_per_step": min(values),
        "max_ns_per_step": max(values),
        "results_ns_per_step": values,
        "spikes": int(statistics.median(spike_counts)) if spike_counts else 0,
        "spike_counts": spike_counts,
    }


def _run_julia_backend() -> dict[str, object]:
    script = f"""
using Statistics
include("src/sc_neurocore/accel/julia/neurons/hay_l5.jl")
const STEPS = {STEPS}
const REPEATS = {REPEATS}
const CURRENT_SOMA = {CURRENT_SOMA}
const CURRENT_TUFT = {CURRENT_TUFT}
function run_once()
    s = HayL5Accel.HayL5PyramidalNeuronState()
    spikes = 0
    start = time_ns()
    for _ in 1:STEPS
        spikes += HayL5Accel.step!(s, CURRENT_SOMA, CURRENT_TUFT)
    end
    elapsed = time_ns() - start
    return elapsed / STEPS, spikes, s.v_s, s.ca_a
end
results = [run_once() for _ in 1:REPEATS]
values = [r[1] for r in results]
println("median_ns_per_step=", median(values))
println("min_ns_per_step=", minimum(values))
println("max_ns_per_step=", maximum(values))
println("results_ns_per_step=", join(values, ","))
println("spike_counts=", join([r[2] for r in results], ","))
println("final_vs=", join([r[3] for r in results], ","))
println("final_cas=", join([r[4] for r in results], ","))
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
        "command": "julia --project=. -e <hay_l5 rk4 benchmark>",
        "steps": STEPS,
        "repeats": len(values),
        "current_soma": CURRENT_SOMA,
        "current_tuft": CURRENT_TUFT,
        "median_ns_per_step": float(fields["median_ns_per_step"]),
        "min_ns_per_step": float(fields["min_ns_per_step"]),
        "max_ns_per_step": float(fields["max_ns_per_step"]),
        "results_ns_per_step": values,
        "spikes": int(fields["spike_counts"].split(",")[0]),
        "final_vs": [float(value) for value in fields["final_vs"].split(",")],
        "final_cas": [float(value) for value in fields["final_cas"].split(",")],
    }


def _run_mojo_backend() -> dict[str, object]:
    command = [
        "mojo",
        "run",
        "--disable-warnings",
        "src/sc_neurocore/accel/mojo/kernels/hay_l5.mojo",
    ]
    try:
        completed = _run_command(command)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        return {"backend": "mojo", "skipped": True, "reason": f"Mojo benchmark failed: {exc}"}
    spikes = int(completed.stdout.rsplit(":", 1)[1].strip())
    return {
        "backend": "mojo",
        "command": " ".join(command),
        "steps": STEPS,
        "current_soma": CURRENT_SOMA,
        "current_tuft": CURRENT_TUFT,
        "spikes": spikes,
    }


def main() -> None:
    report = {
        "benchmark": "hay_l5_rk4",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "steps": STEPS,
        "repeats": REPEATS,
        "current_soma": CURRENT_SOMA,
        "current_tuft": CURRENT_TUFT,
        "expected_spikes": 1,
        "source_hashes": _source_hashes(),
        "backends": [
            _run_python_backend(),
            _run_rust_backend(),
            _run_go_backend(),
            _run_julia_backend(),
            _run_mojo_backend(),
        ],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
