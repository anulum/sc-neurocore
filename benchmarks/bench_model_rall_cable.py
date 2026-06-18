#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Rall cable implicit multi-backend local regression benchmark

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
from typing import cast

from sc_neurocore.neurons.models.rall_cable import RallCableNeuron


STEPS = 200_000
REPEATS = 5
CURRENT = 500.0
OUTPUT = Path("benchmarks/results/local_python_2026-06-18_rall_cable_implicit.json")
REPO_ROOT = Path(__file__).resolve().parents[1]
GO_BENCH_RE = re.compile(r"^BenchmarkRallCableImplicitSolve-\d+\s+\d+\s+([0-9.]+)\s+ns/op")
GO_SPIKES_RE = re.compile(r"\s([0-9.]+)\s+spikes(?:\s|$)")
SOURCE_HASH_PATHS = {
    "benchmarks/bench_model_rall_cable.py": REPO_ROOT / "benchmarks/bench_model_rall_cable.py",
    "engine/Cargo.toml": REPO_ROOT / "engine/Cargo.toml",
    "engine/examples/bench_rall_cable_implicit.rs": REPO_ROOT
    / "engine/examples/bench_rall_cable_implicit.rs",
    "engine/src/neurons/multi_compartment.rs": REPO_ROOT
    / "engine/src/neurons/multi_compartment.rs",
    "src/sc_neurocore/neurons/models/rall_cable.py": REPO_ROOT
    / "src/sc_neurocore/neurons/models/rall_cable.py",
    "src/sc_neurocore/accel/go/services/rall_cable.go": REPO_ROOT
    / "src/sc_neurocore/accel/go/services/rall_cable.go",
    "src/sc_neurocore/accel/go/services/rall_cable_test.go": REPO_ROOT
    / "src/sc_neurocore/accel/go/services/rall_cable_test.go",
    "src/sc_neurocore/accel/julia/neurons/rall_cable.jl": REPO_ROOT
    / "src/sc_neurocore/accel/julia/neurons/rall_cable.jl",
    "src/sc_neurocore/accel/mojo/kernels/rall_cable.mojo": REPO_ROOT
    / "src/sc_neurocore/accel/mojo/kernels/rall_cable.mojo",
    "src/sc_neurocore/accel/rust/safety/rall_cable.rs": REPO_ROOT
    / "src/sc_neurocore/accel/rust/safety/rall_cable.rs",
}


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


def _run_python_backend() -> dict[str, object]:
    results: list[dict[str, object]] = []
    for _ in range(REPEATS):
        neuron = RallCableNeuron(n_comp=5)
        spikes = 0
        start_ns = time.perf_counter_ns()
        for _ in range(STEPS):
            spikes += int(neuron.step(CURRENT))
        elapsed_ns = time.perf_counter_ns() - start_ns
        results.append(
            {
                "backend": "python",
                "steps": STEPS,
                "current": CURRENT,
                "elapsed_ns": elapsed_ns,
                "ns_per_step": elapsed_ns / STEPS,
                "spikes": spikes,
                "ending_state": [float(neuron.v[0]), float(neuron.v[-1])],
            }
        )
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


def _parse_key_value_stdout(stdout: str) -> dict[str, str]:
    return dict(line.split("=", 1) for line in stdout.splitlines() if "=" in line)


def _run_rust_backend() -> dict[str, object]:
    command = [
        "cargo",
        "run",
        "--quiet",
        "--manifest-path",
        "engine/Cargo.toml",
        "--example",
        "bench_rall_cable_implicit",
    ]
    try:
        completed = _run_command(command)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        return {
            "backend": "rust",
            "skipped": True,
            "reason": f"Rust engine benchmark failed: {exc}",
        }
    payload = json.loads(completed.stdout)
    return {
        "backend": "rust",
        "command": " ".join(command),
        "steps": int(payload["steps"]),
        "repeats": int(payload["repeats"]),
        "current": float(payload["current"]),
        "median_ns_per_step": float(payload["median_ns_per_step"]),
        "min_ns_per_step": float(payload["min_ns_per_step"]),
        "max_ns_per_step": float(payload["max_ns_per_step"]),
        "results_ns_per_step": [float(value) for value in payload["results_ns_per_step"]],
        "spikes": int(statistics.median(int(value) for value in payload["spike_counts"])),
        "spike_counts": [int(value) for value in payload["spike_counts"]],
        "final_soma": [float(value) for value in payload["final_soma"]],
        "final_distal": [float(value) for value in payload["final_distal"]],
        "measurement_context": payload["measurement_context"],
    }


def _run_go_backend() -> dict[str, object]:
    command = [
        "go",
        "test",
        "src/sc_neurocore/accel/go/services/rall_cable.go",
        "src/sc_neurocore/accel/go/services/rall_cable_test.go",
        "-run",
        "^$",
        "-bench",
        "BenchmarkRallCableImplicitSolve$",
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
        return {"backend": "go", "skipped": True, "reason": "Go benchmark produced no rows"}
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
include("src/sc_neurocore/accel/julia/neurons/rall_cable.jl")
const STEPS = {STEPS}
const REPEATS = {REPEATS}
const CURRENT = {CURRENT}
function run_once()
    s = RallCableAccel.RallCableNeuronState(5)
    spikes = 0
    start = time_ns()
    for _ in 1:STEPS
        spikes += RallCableAccel.step!(s, CURRENT)
    end
    elapsed = time_ns() - start
    return elapsed / STEPS, spikes, s.v[1], s.v[end]
end
results = [run_once() for _ in 1:REPEATS]
values = [r[1] for r in results]
println("median_ns_per_step=", median(values))
println("min_ns_per_step=", minimum(values))
println("max_ns_per_step=", maximum(values))
println("results_ns_per_step=", join(values, ","))
println("spike_counts=", join([r[2] for r in results], ","))
println("final_soma=", join([r[3] for r in results], ","))
println("final_distal=", join([r[4] for r in results], ","))
"""
    command = ["julia", "--project=.", "-e", script]
    try:
        completed = _run_command(command)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        return {"backend": "julia", "skipped": True, "reason": f"Julia benchmark failed: {exc}"}
    fields = _parse_key_value_stdout(completed.stdout)
    values = [float(value) for value in fields["results_ns_per_step"].split(",")]
    spike_counts = [int(value) for value in fields["spike_counts"].split(",")]
    return {
        "backend": "julia",
        "command": "julia --project=. -e <rall cable implicit benchmark>",
        "steps": STEPS,
        "repeats": len(values),
        "current": CURRENT,
        "median_ns_per_step": float(fields["median_ns_per_step"]),
        "min_ns_per_step": float(fields["min_ns_per_step"]),
        "max_ns_per_step": float(fields["max_ns_per_step"]),
        "results_ns_per_step": values,
        "spikes": int(statistics.median(spike_counts)),
        "spike_counts": spike_counts,
        "final_soma": [float(value) for value in fields["final_soma"].split(",")],
        "final_distal": [float(value) for value in fields["final_distal"].split(",")],
    }


def _run_mojo_backend() -> dict[str, object]:
    program = textwrap.dedent(
        f"""
        from rall_cable import rall_cable_next5, rall_cable_spike
        from std.time import perf_counter

        alias STEPS = {STEPS}
        alias REPEATS = {REPEATS}
        alias CURRENT = {CURRENT}

        def run_once() raises:
            var v0 = -65.0
            var v1 = -65.0
            var v2 = -65.0
            var v3 = -65.0
            var v4 = -65.0
            var spikes = 0
            var start = perf_counter()
            for _ in range(STEPS):
                var n0 = rall_cable_next5(20.0, -65.0, 0.5, 0.1, CURRENT, v0, v1, v2, v3, v4, 0)
                var n1 = rall_cable_next5(20.0, -65.0, 0.5, 0.1, CURRENT, v0, v1, v2, v3, v4, 1)
                var n2 = rall_cable_next5(20.0, -65.0, 0.5, 0.1, CURRENT, v0, v1, v2, v3, v4, 2)
                var n3 = rall_cable_next5(20.0, -65.0, 0.5, 0.1, CURRENT, v0, v1, v2, v3, v4, 3)
                var n4 = rall_cable_next5(20.0, -65.0, 0.5, 0.1, CURRENT, v0, v1, v2, v3, v4, 4)
                var spike = rall_cable_spike(n0, v0, -50.0)
                if spike == 1:
                    n0 = -65.0
                    spikes += 1
                v0 = n0
                v1 = n1
                v2 = n2
                v3 = n3
                v4 = n4
            var elapsed = perf_counter() - start
            print("ns_per_step=", Float64(elapsed) * 1000000000.0 / Float64(STEPS))
            print("spikes=", spikes)
            print("final_soma=", v0)
            print("final_distal=", v4)

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
    final_soma = [
        float(line.split("=", 1)[1])
        for line in completed.stdout.splitlines()
        if line.startswith("final_soma=")
    ]
    final_distal = [
        float(line.split("=", 1)[1])
        for line in completed.stdout.splitlines()
        if line.startswith("final_distal=")
    ]
    if not values:
        return {"backend": "mojo", "skipped": True, "reason": "Mojo benchmark produced no rows"}
    return {
        "backend": "mojo",
        "command": "mojo run --disable-warnings -I src/sc_neurocore/accel/mojo/kernels <temp rall cable benchmark>",
        "steps": STEPS,
        "repeats": len(values),
        "current": CURRENT,
        "median_ns_per_step": statistics.median(values),
        "min_ns_per_step": min(values),
        "max_ns_per_step": max(values),
        "results_ns_per_step": values,
        "spikes": int(statistics.median(spike_counts)),
        "spike_counts": spike_counts,
        "final_soma": final_soma,
        "final_distal": final_distal,
    }


def _backend_summary(payload: dict[str, object]) -> dict[str, object]:
    if payload.get("skipped", False):
        return {"skipped": True, "reason": str(payload.get("reason", "unknown"))}
    return {
        "median_ns_per_step": float(cast(float, payload["median_ns_per_step"])),
        "min_ns_per_step": float(cast(float, payload["min_ns_per_step"])),
        "max_ns_per_step": float(cast(float, payload["max_ns_per_step"])),
        "spikes": int(cast(int, payload["spikes"])),
    }


def _require_all_backends(payloads: list[dict[str, object]]) -> None:
    skipped = [
        f"{payload.get('backend', 'unknown')}: {payload.get('reason', 'unknown')}"
        for payload in payloads
        if payload.get("skipped", False)
    ]
    if skipped:
        raise RuntimeError("Rall cable benchmark requires every backend: " + "; ".join(skipped))


def main() -> None:
    python = _run_python_backend()
    rust = _run_rust_backend()
    go = _run_go_backend()
    julia = _run_julia_backend()
    mojo = _run_mojo_backend()
    _require_all_backends([python, rust, go, julia, mojo])
    payload = {
        "spdx_license": "AGPL-3.0-or-later",
        "benchmark": "RallCableNeuron implicit sealed passive-cable solve",
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "command": "PYTHONPATH=src .venv/bin/python benchmarks/bench_model_rall_cable.py",
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
