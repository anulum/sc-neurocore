#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SpiNNaker LIF exact-flow multi-backend local regression benchmark

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
from typing import Protocol, cast

from sc_neurocore.neurons.models.spinnaker_lif import SpiNNakerLIFNeuron


STEPS = 200_000
REPEATS = 5
CURRENT = 30.0
OUTPUT = Path("benchmarks/results/local_python_2026-06-18_spinnaker_lif_exact_flow.json")
REPO_ROOT = Path(__file__).resolve().parents[1]
GO_BENCH_RE = re.compile(r"^BenchmarkSpiNNakerLIFExactFlow-\d+\s+\d+\s+([0-9.]+)\s+ns/op")
GO_SPIKES_RE = re.compile(r"\s([0-9.]+)\s+spikes(?:\s|$)")
SOURCE_HASH_PATHS = {
    "benchmarks/bench_model_spinnaker_lif.py": REPO_ROOT
    / "benchmarks/bench_model_spinnaker_lif.py",
    "src/sc_neurocore/neurons/models/spinnaker_lif.py": REPO_ROOT
    / "src/sc_neurocore/neurons/models/spinnaker_lif.py",
    "src/sc_neurocore/accel/go/services/spinnaker_lif.go": REPO_ROOT
    / "src/sc_neurocore/accel/go/services/spinnaker_lif.go",
    "src/sc_neurocore/accel/go/services/spinnaker_lif_test.go": REPO_ROOT
    / "src/sc_neurocore/accel/go/services/spinnaker_lif_test.go",
    "src/sc_neurocore/accel/julia/neurons/spinnaker_lif.jl": REPO_ROOT
    / "src/sc_neurocore/accel/julia/neurons/spinnaker_lif.jl",
    "src/sc_neurocore/accel/rust/safety/spinnaker_lif.rs": REPO_ROOT
    / "src/sc_neurocore/accel/rust/safety/spinnaker_lif.rs",
}


class _StepNeuron(Protocol):
    v: float
    refrac_count: float

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


def _run_once(backend: str) -> dict[str, object]:
    neuron: _StepNeuron = SpiNNakerLIFNeuron()
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
        "ending_state": [float(neuron.v), float(neuron.refrac_count)],
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


def _parse_key_value_stdout(stdout: str) -> dict[str, str]:
    return dict(line.split("=", 1) for line in stdout.splitlines() if "=" in line)


def _run_rust_backend() -> dict[str, object]:
    program = textwrap.dedent(
        f"""
        include!(r#"{(REPO_ROOT / "src/sc_neurocore/accel/rust/safety/spinnaker_lif.rs").as_posix()}"#);
        use std::time::Instant;

        const STEPS: usize = {STEPS};
        const REPEATS: usize = {REPEATS};
        const CURRENT: f64 = {CURRENT};

        fn run_once() -> (f64, i32, f64, f64) {{
            let mut neuron = SpiNNakerLIFNeuron::new();
            let mut spikes = 0_i32;
            let start = Instant::now();
            for _ in 0..STEPS {{
                let result = neuron.step(CURRENT);
                if result < 0 {{
                    panic!("invalid exact-flow step");
                }}
                spikes += result;
            }}
            let elapsed_ns = start.elapsed().as_nanos() as f64;
            (elapsed_ns / STEPS as f64, spikes, neuron.v, neuron.refrac_count)
        }}

        fn main() {{
            let mut values = Vec::with_capacity(REPEATS);
            let mut spikes = Vec::with_capacity(REPEATS);
            let mut final_vs = Vec::with_capacity(REPEATS);
            let mut final_refracs = Vec::with_capacity(REPEATS);
            for _ in 0..REPEATS {{
                let (ns, spike_count, v, refrac) = run_once();
                values.push(ns);
                spikes.push(spike_count);
                final_vs.push(v);
                final_refracs.push(refrac);
            }}
            let mut sorted = values.clone();
            sorted.sort_by(|a, b| a.total_cmp(b));
            println!("median_ns_per_step={{}}", sorted[sorted.len() / 2]);
            println!("min_ns_per_step={{}}", sorted[0]);
            println!("max_ns_per_step={{}}", sorted[sorted.len() - 1]);
            println!("results_ns_per_step={{}}", values.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(","));
            println!("spike_counts={{}}", spikes.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(","));
            println!("final_vs={{}}", final_vs.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(","));
            println!("final_refracs={{}}", final_refracs.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(","));
        }}
        """
    )
    with tempfile.TemporaryDirectory() as temp_dir:
        source = Path(temp_dir) / "bench_spinnaker_lif.rs"
        binary = Path(temp_dir) / "bench_spinnaker_lif"
        source.write_text(program, encoding="utf-8")
        try:
            _run_command(["rustc", "-O", str(source), "-o", str(binary)])
            completed = _run_command([str(binary)])
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            return {"backend": "rust", "skipped": True, "reason": f"Rust benchmark failed: {exc}"}
    fields = _parse_key_value_stdout(completed.stdout)
    values = [float(value) for value in fields["results_ns_per_step"].split(",")]
    spike_counts = [int(value) for value in fields["spike_counts"].split(",")]
    return {
        "backend": "rust",
        "command": "rustc -O <temp spinnaker_lif safety benchmark> && <temp spinnaker_lif safety benchmark>",
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
        "final_refracs": [float(value) for value in fields["final_refracs"].split(",")],
    }


def _run_go_backend() -> dict[str, object]:
    command = [
        "go",
        "test",
        "src/sc_neurocore/accel/go/services/spinnaker_lif.go",
        "src/sc_neurocore/accel/go/services/spinnaker_lif_test.go",
        "-run",
        "^$",
        "-bench",
        "BenchmarkSpiNNakerLIFExactFlow$",
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
include("src/sc_neurocore/accel/julia/neurons/spinnaker_lif.jl")
const STEPS = {STEPS}
const REPEATS = {REPEATS}
const CURRENT = {CURRENT}
function run_once()
    s = SpinnakerLifAccel.SpiNNakerLIFNeuronState()
    spikes = 0
    start = time_ns()
    for _ in 1:STEPS
        result = SpinnakerLifAccel.step!(s, CURRENT; dt=s.dt)
        if result < 0
            error("invalid exact-flow step")
        end
        spikes += result
    end
    elapsed = time_ns() - start
    return elapsed / STEPS, spikes, s.v, s.refrac_count
end
results = [run_once() for _ in 1:REPEATS]
values = [r[1] for r in results]
println("median_ns_per_step=", median(values))
println("min_ns_per_step=", minimum(values))
println("max_ns_per_step=", maximum(values))
println("results_ns_per_step=", join(values, ","))
println("spike_counts=", join([r[2] for r in results], ","))
println("final_vs=", join([r[3] for r in results], ","))
println("final_refracs=", join([r[4] for r in results], ","))
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
        "command": "julia --project=. -e <spinnaker_lif exact-flow benchmark>",
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
        "final_refracs": [float(value) for value in fields["final_refracs"].split(",")],
    }


def main() -> int:
    """Run the local multi-backend SpiNNaker LIF benchmark."""

    backend_summary = {
        "python": _run_python_backend(),
        "rust": _run_rust_backend(),
        "go": _run_go_backend(),
        "julia": _run_julia_backend(),
    }
    payload = {
        "spdx_license": "AGPL-3.0-or-later",
        "benchmark": "SpiNNakerLIFNeuron exact constant-current membrane flow",
        "evidence_class": "local_regression_non_isolated",
        "production_speed_claim": False,
        "hardware_measurement_claimed": False,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "steps": STEPS,
        "repeats": REPEATS,
        "current": CURRENT,
        "backend_summary": backend_summary,
        "source_hashes": _source_hashes(),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
