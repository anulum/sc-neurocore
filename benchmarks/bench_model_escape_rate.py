#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Escape-rate exact-flow multi-backend local regression benchmark

from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import platform
import re
import statistics
import subprocess
import tempfile
import textwrap
import time
from typing import Any, Protocol, cast

from sc_neurocore.neurons.models.escape_rate import EscapeRateNeuron


STEPS = 200_000
REPEATS = 5
CURRENT = 30.0
OUTPUT = Path("benchmarks/results/local_python_2026-06-17_escape_rate_exact_flow.json")
REPO_ROOT = Path(__file__).resolve().parents[1]
GO_BENCH_RE = re.compile(r"^BenchmarkEscapeRateExactFlow-\d+\s+\d+\s+([0-9.]+)\s+ns/op")
SOURCE_HASH_PATHS = {
    "benchmarks/bench_model_escape_rate.py": REPO_ROOT / "benchmarks/bench_model_escape_rate.py",
    "engine/Cargo.toml": REPO_ROOT / "engine/Cargo.toml",
    "engine/examples/bench_escape_rate_exact_flow.rs": REPO_ROOT
    / "engine/examples/bench_escape_rate_exact_flow.rs",
    "engine/src/neurons/trivial.rs": REPO_ROOT / "engine/src/neurons/trivial.rs",
    "src/sc_neurocore/neurons/models/escape_rate.py": REPO_ROOT
    / "src/sc_neurocore/neurons/models/escape_rate.py",
    "src/sc_neurocore/accel/go/services/escape_rate.go": REPO_ROOT
    / "src/sc_neurocore/accel/go/services/escape_rate.go",
    "src/sc_neurocore/accel/go/services/escape_rate_test.go": REPO_ROOT
    / "src/sc_neurocore/accel/go/services/escape_rate_test.go",
    "src/sc_neurocore/accel/julia/neurons/escape_rate.jl": REPO_ROOT
    / "src/sc_neurocore/accel/julia/neurons/escape_rate.jl",
    "src/sc_neurocore/accel/mojo/kernels/escape_rate.mojo": REPO_ROOT
    / "src/sc_neurocore/accel/mojo/kernels/escape_rate.mojo",
    "src/sc_neurocore/accel/rust/safety/escape_rate.rs": REPO_ROOT
    / "src/sc_neurocore/accel/rust/safety/escape_rate.rs",
}


class _StepNeuron(Protocol):
    v: float

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
        "ending_state": [float(neuron.v)],
    }


def _run_python_backend() -> dict[str, object]:
    results = [_run_once(lambda: EscapeRateNeuron(), "python") for _ in range(REPEATS)]
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
        "--manifest-path",
        "engine/Cargo.toml",
        "--example",
        "bench_escape_rate_exact_flow",
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
        "src/sc_neurocore/accel/go/services/escape_rate.go",
        "src/sc_neurocore/accel/go/services/escape_rate_test.go",
        "-run",
        "^$",
        "-bench",
        "BenchmarkEscapeRateExactFlow$",
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
            "reason": "Go benchmark output did not include BenchmarkEscapeRateExactFlow ns/op rows",
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
using Random
using Statistics
include("src/sc_neurocore/accel/julia/neurons/escape_rate.jl")
const STEPS = {STEPS}
const REPEATS = {REPEATS}
const CURRENT = {CURRENT}
function run_once(seed)
    Random.seed!(seed)
    s = EscapeRateAccel.EscapeRateNeuronState()
    spikes = 0
    start = time_ns()
    for _ in 1:STEPS
        spikes += EscapeRateAccel.step!(s, CURRENT)
    end
    elapsed = time_ns() - start
    return elapsed / STEPS, spikes, s.v
end
results = [run_once(42 + i) for i in 1:REPEATS]
values = [r[1] for r in results]
println("median_ns_per_step=", median(values))
println("min_ns_per_step=", minimum(values))
println("max_ns_per_step=", maximum(values))
println("results_ns_per_step=", join(values, ","))
println("spike_counts=", join([r[2] for r in results], ","))
println("final_vs=", join([r[3] for r in results], ","))
"""
    command = ["julia", "--project=.", "-e", script]
    try:
        completed = _run_command(command)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        return {"backend": "julia", "skipped": True, "reason": f"Julia benchmark failed: {exc}"}
    fields = dict(line.split("=", 1) for line in completed.stdout.splitlines() if "=" in line)
    values = [float(value) for value in fields["results_ns_per_step"].split(",")]
    spike_counts = [int(value) for value in fields["spike_counts"].split(",")]
    return {
        "backend": "julia",
        "command": "julia --project=. -e <escape-rate exact-flow benchmark>",
        "steps": STEPS,
        "repeats": len(values),
        "current": CURRENT,
        "median_ns_per_step": float(fields["median_ns_per_step"]),
        "min_ns_per_step": float(fields["min_ns_per_step"]),
        "max_ns_per_step": float(fields["max_ns_per_step"]),
        "results_ns_per_step": values,
        "spikes": int(statistics.median(spike_counts)),
        "spike_counts": spike_counts,
        "final_vs": [float(value) for value in fields["final_vs"].split(",")],
    }


def _run_mojo_backend() -> dict[str, object]:
    program = textwrap.dedent(
        f"""
        from escape_rate import escape_rate_next_v, escape_rate_step_spike
        from std.time import perf_counter

        alias STEPS = {STEPS}
        alias REPEATS = {REPEATS}
        alias CURRENT = {CURRENT}

        def run_once() raises:
            var v = -70.0
            var spikes = 0
            var start = perf_counter()
            for i in range(STEPS):
                var threshold = Float64(i % 1000) / 1000.0
                var next_v = escape_rate_next_v(v, CURRENT, -70.0, -70.0, -50.0, 10.0, 0.001, 3.0, 1.0, 1.0)
                var spike = escape_rate_step_spike(v, CURRENT, -70.0, -70.0, -50.0, 10.0, 0.001, 3.0, 1.0, 1.0, threshold)
                if spike == 1:
                    v = -70.0
                    spikes += 1
                else:
                    v = next_v
            var elapsed = perf_counter() - start
            print("ns_per_step=", Float64(elapsed) * 1000000000.0 / Float64(STEPS))
            print("spikes=", spikes)
            print("final_v=", v)

        def main() raises:
            for _ in range(REPEATS):
                run_once()
        """
    )
    with tempfile.NamedTemporaryFile("w", suffix=".mojo", encoding="utf-8") as handle:
        handle.write(program)
        handle.flush()
        command = [
            "mojo",
            "run",
            "--disable-warnings",
            "-I",
            "src/sc_neurocore/accel/mojo/kernels",
            handle.name,
        ]
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
    final_vs = [
        float(line.split("=", 1)[1])
        for line in completed.stdout.splitlines()
        if line.startswith("final_v=")
    ]
    if not values:
        return {"backend": "mojo", "skipped": True, "reason": "Mojo benchmark produced no rows"}
    return {
        "backend": "mojo",
        "command": "mojo run --disable-warnings -I src/sc_neurocore/accel/mojo/kernels <temp escape-rate benchmark>",
        "steps": STEPS,
        "repeats": len(values),
        "current": CURRENT,
        "median_ns_per_step": statistics.median(values),
        "min_ns_per_step": min(values),
        "max_ns_per_step": max(values),
        "results_ns_per_step": values,
        "spikes": int(statistics.median(spike_counts)),
        "spike_counts": spike_counts,
        "final_vs": final_vs,
    }


def _backend_summary(payload: dict[str, object]) -> dict[str, object]:
    if payload.get("skipped", False):
        return {"skipped": True, "reason": str(payload.get("reason", "unknown"))}
    median_ns_per_step = cast(float, payload["median_ns_per_step"])
    min_ns_per_step = cast(float, payload["min_ns_per_step"])
    max_ns_per_step = cast(float, payload["max_ns_per_step"])
    spikes = cast(int, payload["spikes"])
    return {
        "median_ns_per_step": float(median_ns_per_step),
        "min_ns_per_step": float(min_ns_per_step),
        "max_ns_per_step": float(max_ns_per_step),
        "spikes": int(spikes),
    }


def main() -> int:
    python = _run_python_backend()
    rust = _run_rust_backend()
    go = _run_go_backend()
    julia = _run_julia_backend()
    mojo = _run_mojo_backend()
    payload = {
        "spdx_license": "AGPL-3.0-or-later",
        "commercial_license": "available",
        "copyright_concepts": "© Concepts 1996–2026 Miroslav Šotek. All rights reserved.",
        "copyright_code": "© Code 2020–2026 Miroslav Šotek. All rights reserved.",
        "orcid": "0009-0009-3560-0851",
        "contact": "www.anulum.li | protoscience@anulum.li",
        "benchmark": "EscapeRateNeuron exact membrane flow with finite-step escape hazard",
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "command": "PYTHONPATH=src .venv/bin/python benchmarks/bench_model_escape_rate.py",
        "evidence_class": "local_regression_non_isolated",
        "production_speed_claim": False,
        "hardware_measurement_claimed": False,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "steps": STEPS,
        "repeats": REPEATS,
        "current": CURRENT,
        "results": [python, rust, go, julia, mojo],
        "backend_summary": {
            "python": _backend_summary(python),
            "rust": _backend_summary(rust),
            "go": _backend_summary(go),
            "julia": _backend_summary(julia),
            "mojo": _backend_summary(mojo),
        },
        "source_hashes": _source_hashes(),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
