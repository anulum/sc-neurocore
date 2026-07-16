#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Durstewitz D1 PFC neuron RK4 multi-backend local regression benchmark

from __future__ import annotations

import hashlib
import json
import platform
import re
import statistics
import subprocess
import tempfile
import textwrap
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol, cast

from sc_neurocore.accel.mojo.isa_baseline import pin_isa
from sc_neurocore.neurons.models.durstewitz_dopamine import DurstewitzDopamineNeuron

STEPS = 200_000
REPEATS = 5
CURRENT = 10.0
OUTPUT = Path("benchmarks/results/local_python_2026-06-24_durstewitz_dopamine_rk4.json")
REPO_ROOT = Path(__file__).resolve().parents[1]
GO_BENCH_RE = re.compile(r"^BenchmarkDurstewitzRK4-\d+\s+\d+\s+([0-9.]+)\s+ns/op")
GO_SPIKES_RE = re.compile(r"\s([0-9.]+)\s+spikes(?:\s|$)")
SOURCE_HASH_PATHS = {
    "benchmarks/bench_model_durstewitz_dopamine.py": REPO_ROOT
    / "benchmarks/bench_model_durstewitz_dopamine.py",
    "engine/Cargo.toml": REPO_ROOT / "engine/Cargo.toml",
    "engine/examples/bench_durstewitz_dopamine_rk4.rs": REPO_ROOT
    / "engine/examples/bench_durstewitz_dopamine_rk4.rs",
    "engine/src/neurons/biophysical.rs": REPO_ROOT / "engine/src/neurons/biophysical.rs",
    "src/sc_neurocore/neurons/models/durstewitz_dopamine.py": REPO_ROOT
    / "src/sc_neurocore/neurons/models/durstewitz_dopamine.py",
    "src/sc_neurocore/accel/go/services/durstewitz_dopamine.go": REPO_ROOT
    / "src/sc_neurocore/accel/go/services/durstewitz_dopamine.go",
    "src/sc_neurocore/accel/go/services/durstewitz_dopamine_test.go": REPO_ROOT
    / "src/sc_neurocore/accel/go/services/durstewitz_dopamine_test.go",
    "src/sc_neurocore/accel/julia/neurons/durstewitz_dopamine.jl": REPO_ROOT
    / "src/sc_neurocore/accel/julia/neurons/durstewitz_dopamine.jl",
    "src/sc_neurocore/accel/mojo/kernels/durstewitz_dopamine.mojo": REPO_ROOT
    / "src/sc_neurocore/accel/mojo/kernels/durstewitz_dopamine.mojo",
}


class _StepNeuron(Protocol):
    v: float
    h_na: float

    def step(self, current: float) -> int: ...


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_hashes() -> dict[str, str]:
    return {source: _sha256(path) for source, path in SOURCE_HASH_PATHS.items()}


def _run_once(backend: str) -> dict[str, object]:
    neuron: _StepNeuron = DurstewitzDopamineNeuron()
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
        "ending_state": [float(neuron.v), float(neuron.h_na)],
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
        "bench_durstewitz_dopamine_rk4",
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
        "src/sc_neurocore/accel/go/services/durstewitz_dopamine.go",
        "src/sc_neurocore/accel/go/services/durstewitz_dopamine_test.go",
        "-run",
        "^$",
        "-bench",
        "BenchmarkDurstewitzRK4$",
        "-benchtime",
        "200000x",
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
        return {
            "backend": "go",
            "skipped": True,
            "reason": "Go benchmark output did not include BenchmarkDurstewitzRK4 ns/op rows",
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
        "spikes": int(statistics.median(spike_counts)) if spike_counts else 0,
        "spike_counts": spike_counts,
    }


def _run_julia_backend() -> dict[str, object]:
    script = f"""
using Statistics
include("src/sc_neurocore/accel/julia/neurons/durstewitz_dopamine.jl")
const STEPS = {STEPS}
const REPEATS = {REPEATS}
const CURRENT = {CURRENT}
function run_once()
    s = DurstewitzDopamineAccel.DurstewitzDopamineNeuronState()
    spikes = 0
    start = time_ns()
    for _ in 1:STEPS
        spikes += DurstewitzDopamineAccel.step!(s, CURRENT)
    end
    elapsed = time_ns() - start
    return elapsed / STEPS, spikes, s.v, s.h_na
end
results = [run_once() for _ in 1:REPEATS]
values = [r[1] for r in results]
println("median_ns_per_step=", median(values))
println("min_ns_per_step=", minimum(values))
println("max_ns_per_step=", maximum(values))
println("results_ns_per_step=", join(values, ","))
println("spike_counts=", join([r[2] for r in results], ","))
println("final_vs=", join([r[3] for r in results], ","))
println("final_hs=", join([r[4] for r in results], ","))
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
        "command": "julia --project=. -e <durstewitz_dopamine rk4 benchmark>",
        "steps": STEPS,
        "repeats": len(values),
        "current": CURRENT,
        "median_ns_per_step": float(fields["median_ns_per_step"]),
        "min_ns_per_step": float(fields["min_ns_per_step"]),
        "max_ns_per_step": float(fields["max_ns_per_step"]),
        "results_ns_per_step": values,
        "spikes": int(fields["spike_counts"].split(",")[0]),
        "final_vs": [float(value) for value in fields["final_vs"].split(",")],
        "final_hs": [float(value) for value in fields["final_hs"].split(",")],
    }


def _run_mojo_backend() -> dict[str, object]:
    program = textwrap.dedent(
        f"""
        from durstewitz_dopamine import Durstewitz
        from std.time import perf_counter

        alias STEPS = {STEPS}
        alias REPEATS = {REPEATS}
        alias CURRENT = {CURRENT}

        def run_once() raises:
            var neuron = Durstewitz()
            var start = perf_counter()
            var spikes = neuron.simulate(STEPS, CURRENT)
            var elapsed = perf_counter() - start
            print("ns_per_step=", Float64(elapsed) * 1000000000.0 / Float64(STEPS))
            print("spikes=", spikes)

        def main() raises:
            for _ in range(REPEATS):
                run_once()
        """
    )
    with tempfile.NamedTemporaryFile("w", suffix=".mojo", encoding="utf-8") as handle:
        handle.write(program)
        handle.flush()
        command = pin_isa(
            [
                "mojo",
                "run",
                "--disable-warnings",
                "-I",
                "src/sc_neurocore/accel/mojo/kernels",
                handle.name,
            ]
        )
        try:
            completed = _run_command(command)
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            return {"backend": "mojo", "skipped": True, "reason": f"Mojo benchmark failed: {exc}"}
    values = [
        float(line.split("=", 1)[1])
        for line in completed.stdout.splitlines()
        if line.startswith("ns_per_step=")
    ]
    spike_counts = [
        int(line.split("=", 1)[1])
        for line in completed.stdout.splitlines()
        if line.startswith("spikes=")
    ]
    if not values:
        return {"backend": "mojo", "skipped": True, "reason": "Mojo benchmark produced no rows"}
    return {
        "backend": "mojo",
        "command": "mojo run --disable-warnings -I src/sc_neurocore/accel/mojo/kernels <temp durstewitz_dopamine benchmark>",
        "steps": STEPS,
        "repeats": len(values),
        "current": CURRENT,
        "median_ns_per_step": statistics.median(values),
        "min_ns_per_step": min(values),
        "max_ns_per_step": max(values),
        "results_ns_per_step": values,
        "spikes": int(statistics.median(spike_counts)),
        "spike_counts": spike_counts,
    }


def _backend_summary(payload: dict[str, object]) -> dict[str, object]:
    if payload.get("skipped", False):
        return {"skipped": True, "reason": str(payload.get("reason", "unknown"))}
    spikes = payload.get("spikes")
    if not isinstance(spikes, int):
        spike_counts_raw = payload.get("spike_counts")
        if isinstance(spike_counts_raw, list):
            spikes = int(statistics.median([int(value) for value in spike_counts_raw]))
        else:
            spikes = 0
    return {
        "median_ns_per_step": float(cast(float, payload["median_ns_per_step"])),
        "min_ns_per_step": float(cast(float, payload["min_ns_per_step"])),
        "max_ns_per_step": float(cast(float, payload["max_ns_per_step"])),
        "spikes": int(spikes),
    }


def _require_all_backends(payloads: list[dict[str, object]]) -> None:
    skipped = [
        f"{payload.get('backend', 'unknown')}: {payload.get('reason', 'unknown')}"
        for payload in payloads
        if payload.get("skipped", False)
    ]
    if skipped:
        raise RuntimeError("Durstewitz benchmark requires every backend: " + "; ".join(skipped))


def _require_spike_parity(summaries: dict[str, dict[str, object]]) -> None:
    counts = {name: int(cast(int, value["spikes"])) for name, value in summaries.items()}
    if len(set(counts.values())) != 1:
        raise RuntimeError(f"backends disagree on spike count: {counts}")


def main() -> None:
    python = _run_python_backend()
    rust = _run_rust_backend()
    go = _run_go_backend()
    julia = _run_julia_backend()
    mojo = _run_mojo_backend()
    _require_all_backends([python, rust, go, julia, mojo])
    summaries = {
        "python": _backend_summary(python),
        "rust": _backend_summary(rust),
        "go": _backend_summary(go),
        "julia": _backend_summary(julia),
        "mojo": _backend_summary(mojo),
    }
    _require_spike_parity(summaries)
    payload = {
        "spdx_license": "AGPL-3.0-or-later",
        "benchmark": "DurstewitzDopamineNeuron candidate-first RK4 D1-modulated PFC step",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": "PYTHONPATH=src .venv/bin/python benchmarks/bench_model_durstewitz_dopamine.py",
        "evidence_class": "local_regression_non_isolated",
        "production_speed_claim": False,
        "hardware_measurement_claimed": False,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "steps": STEPS,
        "repeats": REPEATS,
        "current": CURRENT,
        "backend_summary": summaries,
        "results": [python, rust, go, julia, mojo],
        "source_hashes": _source_hashes(),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
