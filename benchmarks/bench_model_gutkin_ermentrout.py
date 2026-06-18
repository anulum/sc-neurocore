#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Gutkin-Ermentrout RK4 multi-backend local regression benchmark

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

from sc_neurocore.neurons.models.gutkin_ermentrout import GutkinErmentroutNeuron


STEPS = 200_000
REPEATS = 5
CURRENT = 5.0
OUTPUT = Path("benchmarks/results/local_python_2026-06-18_gutkin_ermentrout_rk4.json")
REPO_ROOT = Path(__file__).resolve().parents[1]
GO_BENCH_RE = re.compile(r"^BenchmarkGutkinErmentroutRK4-\d+\s+\d+\s+([0-9.]+)\s+ns/op")
GO_SPIKES_RE = re.compile(r"\s([0-9.]+)\s+spikes(?:\s|$)")
SOURCE_HASH_PATHS = {
    "benchmarks/bench_model_gutkin_ermentrout.py": REPO_ROOT
    / "benchmarks/bench_model_gutkin_ermentrout.py",
    "engine/Cargo.toml": REPO_ROOT / "engine/Cargo.toml",
    "engine/examples/bench_gutkin_ermentrout_rk4.rs": REPO_ROOT
    / "engine/examples/bench_gutkin_ermentrout_rk4.rs",
    "engine/src/neurons/simple_spiking.rs": REPO_ROOT / "engine/src/neurons/simple_spiking.rs",
    "src/sc_neurocore/neurons/models/gutkin_ermentrout.py": REPO_ROOT
    / "src/sc_neurocore/neurons/models/gutkin_ermentrout.py",
    "src/sc_neurocore/accel/go/services/gutkin_ermentrout.go": REPO_ROOT
    / "src/sc_neurocore/accel/go/services/gutkin_ermentrout.go",
    "src/sc_neurocore/accel/go/services/gutkin_ermentrout_test.go": REPO_ROOT
    / "src/sc_neurocore/accel/go/services/gutkin_ermentrout_test.go",
    "src/sc_neurocore/accel/julia/neurons/gutkin_ermentrout.jl": REPO_ROOT
    / "src/sc_neurocore/accel/julia/neurons/gutkin_ermentrout.jl",
    "src/sc_neurocore/accel/mojo/kernels/gutkin_ermentrout.mojo": REPO_ROOT
    / "src/sc_neurocore/accel/mojo/kernels/gutkin_ermentrout.mojo",
    "src/sc_neurocore/accel/rust/safety/gutkin_ermentrout.rs": REPO_ROOT
    / "src/sc_neurocore/accel/rust/safety/gutkin_ermentrout.rs",
}


class _StepNeuron(Protocol):
    v: float
    n: float

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
        "ending_state": [float(neuron.v), float(neuron.n)],
    }


def _run_python_backend() -> dict[str, object]:
    results = [_run_once(lambda: GutkinErmentroutNeuron(), "python") for _ in range(REPEATS)]
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
        "bench_gutkin_ermentrout_rk4",
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
        "src/sc_neurocore/accel/go/services/gutkin_ermentrout.go",
        "src/sc_neurocore/accel/go/services/gutkin_ermentrout_test.go",
        "-run",
        "^$",
        "-bench",
        "BenchmarkGutkinErmentroutRK4$",
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
            "reason": "Go benchmark output did not include BenchmarkGutkinErmentroutRK4 ns/op rows",
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
include("src/sc_neurocore/accel/julia/neurons/gutkin_ermentrout.jl")
const STEPS = {STEPS}
const REPEATS = {REPEATS}
const CURRENT = {CURRENT}
function run_once()
    s = GutkinErmentroutAccel.GutkinErmentroutNeuronState()
    spikes = 0
    start = time_ns()
    for _ in 1:STEPS
        spikes += GutkinErmentroutAccel.step!(s, CURRENT)
    end
    elapsed = time_ns() - start
    return elapsed / STEPS, spikes, s.v, s.n
end
results = [run_once() for _ in 1:REPEATS]
values = [r[1] for r in results]
println("median_ns_per_step=", median(values))
println("min_ns_per_step=", minimum(values))
println("max_ns_per_step=", maximum(values))
println("results_ns_per_step=", join(values, ","))
println("spike_counts=", join([r[2] for r in results], ","))
println("final_vs=", join([r[3] for r in results], ","))
println("final_ns=", join([r[4] for r in results], ","))
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
        "command": "julia --project=. -e <gutkin-ermentrout rk4 benchmark>",
        "steps": STEPS,
        "repeats": len(values),
        "current": CURRENT,
        "median_ns_per_step": float(fields["median_ns_per_step"]),
        "min_ns_per_step": float(fields["min_ns_per_step"]),
        "max_ns_per_step": float(fields["max_ns_per_step"]),
        "results_ns_per_step": values,
        "spikes": int(fields["spike_counts"].split(",")[0]),
        "final_vs": [float(value) for value in fields["final_vs"].split(",")],
        "final_ns": [float(value) for value in fields["final_ns"].split(",")],
    }


def _run_mojo_backend() -> dict[str, object]:
    program = textwrap.dedent(
        f"""
        from gutkin_ermentrout import gutkin_ermentrout_next_n, gutkin_ermentrout_next_v, gutkin_ermentrout_step_spike
        from std.time import perf_counter

        alias STEPS = {STEPS}
        alias REPEATS = {REPEATS}
        alias CURRENT = {CURRENT}

        def run_once() raises:
            var v = -65.0
            var n = 0.1
            var spikes = 0
            var start = perf_counter()
            for _ in range(STEPS):
                var next_v = gutkin_ermentrout_next_v(v, n, CURRENT, 20.0, 10.0, 8.0, 60.0, -90.0, -80.0, 0.05, -20.0)
                var next_n = gutkin_ermentrout_next_n(v, n, CURRENT, 20.0, 10.0, 8.0, 60.0, -90.0, -80.0, 0.05, -20.0)
                spikes += gutkin_ermentrout_step_spike(v, next_v, -20.0)
                v = next_v
                n = next_n
            var elapsed = perf_counter() - start
            print("ns_per_step=", Float64(elapsed) * 1000000000.0 / Float64(STEPS))
            print("spikes=", spikes)
            print("final_v=", v)
            print("final_n=", n)

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
    final_ns = [
        float(line.split("=", 1)[1])
        for line in completed.stdout.splitlines()
        if line.startswith("final_n=")
    ]
    if not values:
        return {"backend": "mojo", "skipped": True, "reason": "Mojo benchmark produced no rows"}
    return {
        "backend": "mojo",
        "command": "mojo run --disable-warnings -I src/sc_neurocore/accel/mojo/kernels <temp gutkin-ermentrout benchmark>",
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
        "final_ns": final_ns,
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
        raise RuntimeError(
            "Gutkin-Ermentrout benchmark requires every backend: " + "; ".join(skipped)
        )


def main() -> None:
    python = _run_python_backend()
    rust = _run_rust_backend()
    go = _run_go_backend()
    julia = _run_julia_backend()
    mojo = _run_mojo_backend()
    _require_all_backends([python, rust, go, julia, mojo])
    payload = {
        "spdx_license": "AGPL-3.0-or-later",
        "benchmark": "GutkinErmentroutNeuron candidate-first RK4 conductance step",
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "command": "PYTHONPATH=src .venv/bin/python benchmarks/bench_model_gutkin_ermentrout.py",
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
            "python": _backend_summary(python),
            "rust": _backend_summary(rust),
            "go": _backend_summary(go),
            "julia": _backend_summary(julia),
            "mojo": _backend_summary(mojo),
        },
        "results": [python, rust, go, julia, mojo],
        "source_hashes": _source_hashes(),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
