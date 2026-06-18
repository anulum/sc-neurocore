#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SRM0 exact-flow multi-backend local regression benchmark

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

from sc_neurocore.neurons.models.srm0 import SRM0Neuron


STEPS = 200_000
REPEATS = 5
CURRENT = 2.0
OUTPUT = Path("benchmarks/results/local_python_2026-06-18_srm0_exact_flow.json")
REPO_ROOT = Path(__file__).resolve().parents[1]
GO_BENCH_RE = re.compile(r"^BenchmarkSRM0ExactFlow-\d+\s+\d+\s+([0-9.]+)\s+ns/op")
GO_SPIKES_RE = re.compile(r"\s([0-9.]+)\s+spikes(?:\s|$)")
SOURCE_HASH_PATHS = {
    "benchmarks/bench_model_srm0.py": REPO_ROOT / "benchmarks/bench_model_srm0.py",
    "src/sc_neurocore/neurons/models/srm0.py": REPO_ROOT
    / "src/sc_neurocore/neurons/models/srm0.py",
    "src/sc_neurocore/accel/go/services/srm0.go": REPO_ROOT
    / "src/sc_neurocore/accel/go/services/srm0.go",
    "src/sc_neurocore/accel/go/services/srm0_test.go": REPO_ROOT
    / "src/sc_neurocore/accel/go/services/srm0_test.go",
    "src/sc_neurocore/accel/julia/neurons/srm0.jl": REPO_ROOT
    / "src/sc_neurocore/accel/julia/neurons/srm0.jl",
    "src/sc_neurocore/accel/mojo/kernels/srm0.mojo": REPO_ROOT
    / "src/sc_neurocore/accel/mojo/kernels/srm0.mojo",
    "src/sc_neurocore/accel/rust/safety/srm0.rs": REPO_ROOT
    / "src/sc_neurocore/accel/rust/safety/srm0.rs",
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
    results = [_run_once(lambda: SRM0Neuron(), "python") for _ in range(REPEATS)]
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
    program = textwrap.dedent(
        f"""
        mod srm0 {{
            include!(r#"{(REPO_ROOT / "src/sc_neurocore/accel/rust/safety/srm0.rs").as_posix()}"#);
        }}
        use srm0::SRM0Neuron;
        use std::time::Instant;

        const STEPS: usize = {STEPS};
        const REPEATS: usize = {REPEATS};
        const CURRENT: f64 = {CURRENT};

        fn run_once() -> (f64, i32, f64, f64) {{
            let mut neuron = SRM0Neuron::default();
            let mut spikes = 0_i32;
            let start = Instant::now();
            for _ in 0..STEPS {{
                spikes += neuron.step(CURRENT);
            }}
            let elapsed_ns = start.elapsed().as_nanos() as f64;
            (elapsed_ns / STEPS as f64, spikes, neuron.v, neuron.eta)
        }}

        fn main() {{
            let mut values = Vec::with_capacity(REPEATS);
            let mut spikes = Vec::with_capacity(REPEATS);
            let mut final_vs = Vec::with_capacity(REPEATS);
            let mut final_etas = Vec::with_capacity(REPEATS);
            for _ in 0..REPEATS {{
                let (ns, spike_count, v, eta) = run_once();
                values.push(ns);
                spikes.push(spike_count);
                final_vs.push(v);
                final_etas.push(eta);
            }}
            let mut sorted = values.clone();
            sorted.sort_by(|a, b| a.total_cmp(b));
            println!("median_ns_per_step={{}}", sorted[sorted.len() / 2]);
            println!("min_ns_per_step={{}}", sorted[0]);
            println!("max_ns_per_step={{}}", sorted[sorted.len() - 1]);
            println!("results_ns_per_step={{}}", values.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(","));
            println!("spike_counts={{}}", spikes.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(","));
            println!("final_vs={{}}", final_vs.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(","));
            println!("final_etas={{}}", final_etas.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(","));
        }}
        """
    )
    with tempfile.TemporaryDirectory() as temp_dir:
        source = Path(temp_dir) / "bench_srm0.rs"
        binary = Path(temp_dir) / "bench_srm0"
        source.write_text(program, encoding="utf-8")
        command = ["rustc", "-O", str(source), "-o", str(binary)]
        try:
            _run_command(command)
            completed = _run_command([str(binary)])
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            return {"backend": "rust", "skipped": True, "reason": f"Rust benchmark failed: {exc}"}
    fields = _parse_key_value_stdout(completed.stdout)
    values = [float(value) for value in fields["results_ns_per_step"].split(",")]
    spike_counts = [int(value) for value in fields["spike_counts"].split(",")]
    return {
        "backend": "rust",
        "command": "rustc -O <temp srm0 safety benchmark> && <temp srm0 safety benchmark>",
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
        "final_etas": [float(value) for value in fields["final_etas"].split(",")],
    }


def _run_go_backend() -> dict[str, object]:
    command = [
        "go",
        "test",
        "src/sc_neurocore/accel/go/services/srm0.go",
        "src/sc_neurocore/accel/go/services/srm0_test.go",
        "-run",
        "^$",
        "-bench",
        "BenchmarkSRM0ExactFlow$",
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
include("src/sc_neurocore/accel/julia/neurons/srm0.jl")
const STEPS = {STEPS}
const REPEATS = {REPEATS}
const CURRENT = {CURRENT}
function run_once()
    s = Srm0Accel.SRM0NeuronState()
    spikes = 0
    start = time_ns()
    for _ in 1:STEPS
        spikes += Srm0Accel.step!(s, CURRENT)
    end
    elapsed = time_ns() - start
    return elapsed / STEPS, spikes, s.v, s.eta
end
results = [run_once() for _ in 1:REPEATS]
values = [r[1] for r in results]
println("median_ns_per_step=", median(values))
println("min_ns_per_step=", minimum(values))
println("max_ns_per_step=", maximum(values))
println("results_ns_per_step=", join(values, ","))
println("spike_counts=", join([r[2] for r in results], ","))
println("final_vs=", join([r[3] for r in results], ","))
println("final_etas=", join([r[4] for r in results], ","))
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
        "command": "julia --project=. -e <srm0 exact-flow benchmark>",
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
        "final_etas": [float(value) for value in fields["final_etas"].split(",")],
    }


def _run_mojo_backend() -> dict[str, object]:
    program = textwrap.dedent(
        f"""
        from srm0 import srm0_next_eta, srm0_next_v, srm0_step_spike
        from std.time import perf_counter

        alias STEPS = {STEPS}
        alias REPEATS = {REPEATS}
        alias CURRENT = {CURRENT}

        def run_once() raises:
            var v = 0.0
            var eta = 0.0
            var t = 0.0
            var spikes = 0
            var start = perf_counter()
            for _ in range(STEPS):
                var next_v = srm0_next_v(v, 0.0, 1.0, 20.0, 50.0, 5.0, 1.0, 1.0, eta, t, CURRENT)
                var spike = srm0_step_spike(next_v, 1.0)
                t += 1.0
                if spike == 1:
                    v = 0.0
                    eta = -5.0
                    spikes += 1
                else:
                    v = next_v
                    eta = srm0_next_eta(eta, 50.0, 1.0)
            var elapsed = perf_counter() - start
            print("ns_per_step=", Float64(elapsed) * 1000000000.0 / Float64(STEPS))
            print("spikes=", spikes)
            print("final_v=", v)
            print("final_eta=", eta)

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
    final_etas = [
        float(line.split("=", 1)[1])
        for line in completed.stdout.splitlines()
        if line.startswith("final_eta=")
    ]
    if not values:
        return {"backend": "mojo", "skipped": True, "reason": "Mojo benchmark produced no rows"}
    return {
        "backend": "mojo",
        "command": "mojo run --disable-warnings -I src/sc_neurocore/accel/mojo/kernels <temp srm0 benchmark>",
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
        "final_etas": final_etas,
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
        raise RuntimeError("SRM0 benchmark requires every backend: " + "; ".join(skipped))


def main() -> None:
    python = _run_python_backend()
    rust = _run_rust_backend()
    go = _run_go_backend()
    julia = _run_julia_backend()
    mojo = _run_mojo_backend()
    _require_all_backends([python, rust, go, julia, mojo])
    payload = {
        "spdx_license": "AGPL-3.0-or-later",
        "benchmark": "SRM0Neuron exact coupled refractory-kernel flow step",
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "command": "PYTHONPATH=src .venv/bin/python benchmarks/bench_model_srm0.py",
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
