#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Butera Model 1 source RK4 multi-backend benchmark

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
from typing import cast

from sc_neurocore.accel.mojo.isa_baseline import pin_isa
from sc_neurocore.neurons.models.butera_respiratory import ButeraRespiratoryNeuron

STEPS = 200_000
REPEATS = 5
CURRENT = 50.0
OUTPUT = Path("benchmarks/results/bench_butera_respiratory.json")
REPO_ROOT = Path(__file__).resolve().parents[1]
GO_BENCH_RE = re.compile(r"^BenchmarkButeraRespiratoryRK4(?:-\d+)?\s+\d+\s+([0-9.]+)\s+ns/op")
GO_SPIKES_RE = re.compile(r"\s([0-9.]+)\s+spikes(?:\s|$)")
SOURCE_HASH_PATHS = {
    "benchmarks/bench_model_butera_respiratory.py": REPO_ROOT
    / "benchmarks/bench_model_butera_respiratory.py",
    "engine/Cargo.toml": REPO_ROOT / "engine/Cargo.toml",
    "engine/examples/bench_butera_respiratory_rk4.rs": REPO_ROOT
    / "engine/examples/bench_butera_respiratory_rk4.rs",
    "engine/src/neurons/simple_spiking.rs": REPO_ROOT / "engine/src/neurons/simple_spiking.rs",
    "engine/src/neurons/simple_spiking/reexports.rs": REPO_ROOT
    / "engine/src/neurons/simple_spiking/reexports.rs",
    "engine/src/neurons/simple_spiking/butera_respiratory.rs": REPO_ROOT
    / "engine/src/neurons/simple_spiking/butera_respiratory.rs",
    "src/sc_neurocore/neurons/models/butera_respiratory.py": REPO_ROOT
    / "src/sc_neurocore/neurons/models/butera_respiratory.py",
    "src/sc_neurocore/accel/go/services/butera_respiratory.go": REPO_ROOT
    / "src/sc_neurocore/accel/go/services/butera_respiratory.go",
    "src/sc_neurocore/accel/go/services/butera_respiratory_test.go": REPO_ROOT
    / "src/sc_neurocore/accel/go/services/butera_respiratory_test.go",
    "src/sc_neurocore/accel/julia/neurons/butera_respiratory.jl": REPO_ROOT
    / "src/sc_neurocore/accel/julia/neurons/butera_respiratory.jl",
    "src/sc_neurocore/accel/mojo/kernels/butera_respiratory.mojo": REPO_ROOT
    / "src/sc_neurocore/accel/mojo/kernels/butera_respiratory.mojo",
    "src/sc_neurocore/accel/rust/safety/butera_respiratory.rs": REPO_ROOT
    / "src/sc_neurocore/accel/rust/safety/butera_respiratory.rs",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_hashes() -> dict[str, str]:
    return {source: _sha256(path) for source, path in SOURCE_HASH_PATHS.items()}


def _run_python_once() -> dict[str, object]:
    neuron = ButeraRespiratoryNeuron()
    spikes = 0
    started = time.perf_counter_ns()
    for _ in range(STEPS):
        spikes += neuron.step(CURRENT)
    elapsed = time.perf_counter_ns() - started
    return {
        "ns_per_step": elapsed / STEPS,
        "spikes": spikes,
        "final_state": [neuron.v, neuron.n, neuron.h_nap],
    }


def _run_python() -> dict[str, object]:
    results = [_run_python_once() for _ in range(REPEATS)]
    values = [cast(float, result["ns_per_step"]) for result in results]
    return {
        "backend": "python",
        "median_ns_per_step": statistics.median(values),
        "min_ns_per_step": min(values),
        "max_ns_per_step": max(values),
        "spikes": cast(int, results[0]["spikes"]),
        "results": results,
    }


def _command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, text=True, capture_output=True)


def _run_rust() -> dict[str, object]:
    command = [
        "cargo",
        "run",
        "--release",
        "--no-default-features",
        "--manifest-path",
        "engine/Cargo.toml",
        "--example",
        "bench_butera_respiratory_rk4",
    ]
    report = cast(dict[str, object], json.loads(_command(command).stdout))
    spike_counts = cast(list[int], report["spike_counts"])
    report["spikes"] = spike_counts[0]
    return report


def _run_go() -> dict[str, object]:
    command = [
        "go",
        "test",
        "src/sc_neurocore/accel/go/services/butera_respiratory.go",
        "src/sc_neurocore/accel/go/services/butera_respiratory_test.go",
        "src/sc_neurocore/accel/go/services/sc_unit_capacitance_respiratory.go",
        "-run",
        "^$",
        "-bench",
        "BenchmarkButeraRespiratoryRK4$",
        "-benchtime",
        f"{STEPS}x",
        "-count",
        str(REPEATS),
    ]
    completed = _command(command)
    values: list[float] = []
    spike_counts: list[int] = []
    for line in completed.stdout.splitlines():
        if match := GO_BENCH_RE.match(line):
            values.append(float(match.group(1)))
            if spike_match := GO_SPIKES_RE.search(line):
                spike_counts.append(int(float(spike_match.group(1))))
    if len(values) != REPEATS or len(spike_counts) != REPEATS:
        raise RuntimeError(f"incomplete Go benchmark output:\n{completed.stdout}")
    return {
        "backend": "go",
        "command": " ".join(command),
        "median_ns_per_step": statistics.median(values),
        "min_ns_per_step": min(values),
        "max_ns_per_step": max(values),
        "results_ns_per_step": values,
        "spikes": spike_counts[0],
        "spike_counts": spike_counts,
    }


def _run_julia() -> dict[str, object]:
    script = f"""
using Statistics
include("src/sc_neurocore/accel/julia/neurons/butera_respiratory.jl")
function run_once()
    s = ButeraRespiratoryAccel.ButeraRespiratoryNeuronState()
    spikes = 0
    started = time_ns()
    for _ in 1:{STEPS}
        spikes += ButeraRespiratoryAccel.step!(s, {CURRENT})
    end
    return (time_ns() - started) / {STEPS}, spikes, s.v, s.n, s.h_nap
end
results = [run_once() for _ in 1:{REPEATS}]
values = [result[1] for result in results]
println("values=", join(values, ","))
println("spikes=", join([result[2] for result in results], ","))
println("states=", join([join(result[3:5], ":") for result in results], ","))
"""
    command = ["julia", "--startup-file=no", "-e", script]
    fields = dict(
        line.split("=", 1) for line in _command(command).stdout.splitlines() if "=" in line
    )
    values = [float(value) for value in fields["values"].split(",")]
    spike_counts = [int(value) for value in fields["spikes"].split(",")]
    return {
        "backend": "julia",
        "command": "julia --startup-file=no -e <butera Model 1 benchmark>",
        "median_ns_per_step": statistics.median(values),
        "min_ns_per_step": min(values),
        "max_ns_per_step": max(values),
        "results_ns_per_step": values,
        "spikes": spike_counts[0],
        "spike_counts": spike_counts,
        "final_states": fields["states"],
    }


def _run_mojo() -> dict[str, object]:
    program = textwrap.dedent(
        f"""
        from butera_respiratory import ButeraRespiratory
        from std.time import perf_counter

        comptime STEPS = {STEPS}
        comptime REPEATS = {REPEATS}
        comptime CURRENT = {CURRENT}

        def main() raises:
            for _ in range(REPEATS):
                var neuron = ButeraRespiratory()
                var started = perf_counter()
                var spikes = neuron.simulate(STEPS, CURRENT)
                var elapsed = perf_counter() - started
                print("ns=", Float64(elapsed) * 1000000000.0 / Float64(STEPS))
                print("spikes=", spikes)
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
        lines = _command(command).stdout.splitlines()
    values = [float(line.split("=", 1)[1]) for line in lines if line.startswith("ns=")]
    spike_counts = [int(line.split("=", 1)[1]) for line in lines if line.startswith("spikes=")]
    if len(values) != REPEATS or len(spike_counts) != REPEATS:
        raise RuntimeError(f"incomplete Mojo benchmark output: {lines}")
    return {
        "backend": "mojo",
        "command": "mojo run --disable-warnings -I <kernels> <temporary benchmark>",
        "median_ns_per_step": statistics.median(values),
        "min_ns_per_step": min(values),
        "max_ns_per_step": max(values),
        "results_ns_per_step": values,
        "spikes": spike_counts[0],
        "spike_counts": spike_counts,
    }


def main() -> None:
    backends = {
        result["backend"]: result
        for result in (_run_python(), _run_rust(), _run_go(), _run_julia(), _run_mojo())
    }
    spike_counts = {cast(int, result["spikes"]) for result in backends.values()}
    if len(spike_counts) != 1:
        raise RuntimeError(f"backend event mismatch: {spike_counts}")
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "system": platform.platform(),
        "python": platform.python_version(),
        "steps": STEPS,
        "repeats": REPEATS,
        "current": CURRENT,
        "backend_results": backends,
        "backend_summary": {
            name: {
                "median_ns_per_step": result["median_ns_per_step"],
                "min_ns_per_step": result["min_ns_per_step"],
                "max_ns_per_step": result["max_ns_per_step"],
                "spikes": result["spikes"],
            }
            for name, result in backends.items()
        },
        "source_hashes": _source_hashes(),
        "production_speed_claim": False,
        "hardware_measurement_claimed": False,
        "notes": "Local loaded-host regression only; source event parity is the stable cross-runtime observable.",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "backend_summary": payload["backend_summary"],
                "source_hashes": payload["source_hashes"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
