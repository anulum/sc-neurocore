#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — retained unit-capacitance respiratory multi-backend benchmark

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
from sc_neurocore.neurons.models.sc_unit_capacitance_respiratory import (
    SCUnitCapacitanceRespiratoryNeuron,
)

STEPS = 20_000
REPEATS = 5
CURRENT = 20.0
OUTPUT = Path("benchmarks/results/bench_sc_unit_capacitance_respiratory.json")
REPO_ROOT = Path(__file__).resolve().parents[1]
GO_RE = re.compile(r"^BenchmarkSCUnitCapacitanceRespiratoryRK4(?:-\d+)?\s+\d+\s+([0-9.]+)\s+ns/op")
GO_SPIKES_RE = re.compile(r"\s([0-9.]+)\s+spikes(?:\s|$)")
SOURCE_HASH_PATHS = {
    "benchmarks/bench_model_sc_unit_capacitance_respiratory.py": Path(__file__).resolve(),
    "engine/Cargo.toml": REPO_ROOT / "engine/Cargo.toml",
    "engine/examples/bench_sc_unit_capacitance_respiratory_rk4.rs": REPO_ROOT
    / "engine/examples/bench_sc_unit_capacitance_respiratory_rk4.rs",
    "engine/src/neurons/simple_spiking.rs": REPO_ROOT / "engine/src/neurons/simple_spiking.rs",
    "engine/src/neurons/simple_spiking/reexports.rs": REPO_ROOT
    / "engine/src/neurons/simple_spiking/reexports.rs",
    "engine/src/neurons/simple_spiking/butera_respiratory.rs": REPO_ROOT
    / "engine/src/neurons/simple_spiking/butera_respiratory.rs",
    "engine/src/neurons/simple_spiking/sc_unit_capacitance_respiratory.rs": REPO_ROOT
    / "engine/src/neurons/simple_spiking/sc_unit_capacitance_respiratory.rs",
    "src/sc_neurocore/neurons/models/butera_respiratory.py": REPO_ROOT
    / "src/sc_neurocore/neurons/models/butera_respiratory.py",
    "src/sc_neurocore/neurons/models/sc_unit_capacitance_respiratory.py": REPO_ROOT
    / "src/sc_neurocore/neurons/models/sc_unit_capacitance_respiratory.py",
    "src/sc_neurocore/accel/go/services/butera_respiratory.go": REPO_ROOT
    / "src/sc_neurocore/accel/go/services/butera_respiratory.go",
    "src/sc_neurocore/accel/go/services/butera_respiratory_test.go": REPO_ROOT
    / "src/sc_neurocore/accel/go/services/butera_respiratory_test.go",
    "src/sc_neurocore/accel/go/services/sc_unit_capacitance_respiratory.go": REPO_ROOT
    / "src/sc_neurocore/accel/go/services/sc_unit_capacitance_respiratory.go",
    "src/sc_neurocore/accel/julia/neurons/butera_respiratory.jl": REPO_ROOT
    / "src/sc_neurocore/accel/julia/neurons/butera_respiratory.jl",
    "src/sc_neurocore/accel/julia/neurons/sc_unit_capacitance_respiratory.jl": REPO_ROOT
    / "src/sc_neurocore/accel/julia/neurons/sc_unit_capacitance_respiratory.jl",
    "src/sc_neurocore/accel/mojo/kernels/butera_respiratory.mojo": REPO_ROOT
    / "src/sc_neurocore/accel/mojo/kernels/butera_respiratory.mojo",
    "src/sc_neurocore/accel/mojo/kernels/sc_unit_capacitance_respiratory.mojo": REPO_ROOT
    / "src/sc_neurocore/accel/mojo/kernels/sc_unit_capacitance_respiratory.mojo",
    "src/sc_neurocore/accel/rust/safety/butera_respiratory.rs": REPO_ROOT
    / "src/sc_neurocore/accel/rust/safety/butera_respiratory.rs",
    "src/sc_neurocore/accel/rust/safety/sc_unit_capacitance_respiratory.rs": REPO_ROOT
    / "src/sc_neurocore/accel/rust/safety/sc_unit_capacitance_respiratory.rs",
}


def _command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, text=True, capture_output=True)


def _summary(
    name: str, values: list[float], spikes: list[int], **extra: object
) -> dict[str, object]:
    return {
        "backend": name,
        "median_ns_per_step": statistics.median(values),
        "min_ns_per_step": min(values),
        "max_ns_per_step": max(values),
        "results_ns_per_step": values,
        "spikes": spikes[0],
        "spike_counts": spikes,
        **extra,
    }


def _run_python() -> dict[str, object]:
    values: list[float] = []
    spikes: list[int] = []
    states: list[list[float]] = []
    for _ in range(REPEATS):
        neuron = SCUnitCapacitanceRespiratoryNeuron()
        count = 0
        started = time.perf_counter_ns()
        for _ in range(STEPS):
            count += neuron.step(CURRENT)
        values.append((time.perf_counter_ns() - started) / STEPS)
        spikes.append(count)
        states.append([neuron.v, neuron.n, neuron.h_nap])
    return _summary("python", values, spikes, final_states=states)


def _run_rust() -> dict[str, object]:
    command = [
        "cargo",
        "run",
        "--release",
        "--no-default-features",
        "--manifest-path",
        "engine/Cargo.toml",
        "--example",
        "bench_sc_unit_capacitance_respiratory_rk4",
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
        "BenchmarkSCUnitCapacitanceRespiratoryRK4$",
        "-benchtime",
        f"{STEPS}x",
        "-count",
        str(REPEATS),
    ]
    lines = _command(command).stdout.splitlines()
    values = [float(match.group(1)) for line in lines if (match := GO_RE.match(line))]
    spikes = [int(float(match.group(1))) for line in lines if (match := GO_SPIKES_RE.search(line))]
    if len(values) != REPEATS or len(spikes) != REPEATS:
        raise RuntimeError(f"incomplete Go benchmark output: {lines}")
    return _summary("go", values, spikes, command=" ".join(command))


def _run_julia() -> dict[str, object]:
    script = f"""
using Statistics
include("src/sc_neurocore/accel/julia/neurons/sc_unit_capacitance_respiratory.jl")
const SC = SCUnitCapacitanceRespiratoryAccel
function run_once()
    state = SC.SCUnitCapacitanceRespiratoryNeuronState()
    spikes = 0
    started = time_ns()
    for _ in 1:{STEPS}
        spikes += SC.step!(state, {CURRENT})
    end
    return (time_ns()-started)/{STEPS}, spikes, state.v, state.n, state.h_nap
end
results = [run_once() for _ in 1:{REPEATS}]
println("values=", join([r[1] for r in results], ","))
println("spikes=", join([r[2] for r in results], ","))
println("states=", join([join(r[3:5], ":") for r in results], ","))
"""
    lines = _command(["julia", "--startup-file=no", "-e", script]).stdout.splitlines()
    fields = dict(line.split("=", 1) for line in lines if "=" in line)
    values = [float(value) for value in fields["values"].split(",")]
    spikes = [int(value) for value in fields["spikes"].split(",")]
    return _summary("julia", values, spikes, final_states=fields["states"])


def _run_mojo() -> dict[str, object]:
    program = textwrap.dedent(
        f"""
        from sc_unit_capacitance_respiratory import SCUnitCapacitanceRespiratory
        from std.time import perf_counter

        comptime STEPS = {STEPS}
        comptime REPEATS = {REPEATS}

        def main() raises:
            for _ in range(REPEATS):
                var neuron = SCUnitCapacitanceRespiratory()
                var started = perf_counter()
                var spikes = neuron.simulate(STEPS, {CURRENT})
                var elapsed = perf_counter() - started
                print("ns=", Float64(elapsed)*1000000000.0/Float64(STEPS))
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
    spikes = [int(line.split("=", 1)[1]) for line in lines if line.startswith("spikes=")]
    if len(values) != REPEATS or len(spikes) != REPEATS:
        raise RuntimeError(f"incomplete Mojo benchmark output: {lines}")
    return _summary("mojo", values, spikes, command="mojo run <SC benchmark>")


def main() -> None:
    results = [_run_python(), _run_rust(), _run_go(), _run_julia(), _run_mojo()]
    counts = {str(result["backend"]): int(cast(int, result["spikes"])) for result in results}
    if max(counts.values()) - min(counts.values()) > 1:
        raise RuntimeError(f"SC event envelope diverged: {counts}")
    payload = {
        "spdx_license": "AGPL-3.0-or-later",
        "benchmark": "SCUnitCapacitanceRespiratoryNeuron retained RK4 profile",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": "PYTHONPATH=src .venv/bin/python benchmarks/bench_model_sc_unit_capacitance_respiratory.py",
        "evidence_class": "local_regression_non_isolated",
        "production_speed_claim": False,
        "hardware_measurement_claimed": False,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "steps": STEPS,
        "repeats": REPEATS,
        "current": CURRENT,
        "event_contract": "Python/Rust/Go/Julia=5; Mojo=4 accepted only with one-step state parity and maximum event-count spread 1",
        "results": results,
        "source_hashes": {
            source: hashlib.sha256(path.read_bytes()).hexdigest()
            for source, path in SOURCE_HASH_PATHS.items()
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
