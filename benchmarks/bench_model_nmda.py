#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — NMDA dual-identity five-runtime benchmark

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import statistics
import subprocess
import tempfile
import textwrap
import time

from sc_neurocore.accel.mojo.isa_baseline import pin_isa
from sc_neurocore.neurons.models import NMDANeuron, SCWBNMDAMagnesiumBlockNeuron

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "benchmarks/results/bench_nmda.json"
STEPS = 20_000
REPEATS = 3
SOURCES = [
    "benchmarks/bench_model_nmda.py",
    "engine/examples/bench_nmda_rk2.rs",
    "engine/src/neurons/channels/nmda.rs",
    "engine/src/neurons/channels/sc_wb_nmda_magnesium_block.rs",
    "src/sc_neurocore/neurons/models/nmda_neuron.py",
    "src/sc_neurocore/neurons/models/sc_wb_nmda_magnesium_block.py",
    "src/sc_neurocore/accel/go/services/nmda_neuron.go",
    "src/sc_neurocore/accel/go/services/sc_wb_nmda_magnesium_block.go",
    "src/sc_neurocore/accel/julia/neurons/nmda_neuron.jl",
    "src/sc_neurocore/accel/julia/neurons/sc_wb_nmda_magnesium_block.jl",
    "src/sc_neurocore/accel/mojo/kernels/nmda_neuron.mojo",
    "src/sc_neurocore/accel/mojo/kernels/sc_wb_nmda_magnesium_block.mojo",
    "src/sc_neurocore/accel/rust/safety/nmda_neuron.rs",
    "src/sc_neurocore/accel/rust/safety/sc_wb_nmda_magnesium_block.rs",
]


def _summary(
    source: list[float], sc: list[float], source_state: list[float], sc_state: list[float]
) -> dict[str, object]:
    return {
        "source_median_ns_per_step": statistics.median(source),
        "sc_median_ns_per_step": statistics.median(sc),
        "source_results_ns_per_step": source,
        "sc_results_ns_per_step": sc,
        "source_final_state": source_state,
        "sc_final_state": sc_state,
    }


def _python() -> dict[str, object]:
    source_times: list[float] = []
    sc_times: list[float] = []
    source_state: list[float] = []
    sc_state: list[float] = []
    for _ in range(REPEATS):
        source = NMDANeuron()
        started = time.perf_counter_ns()
        for _ in range(STEPS):
            source.step(0.6)
        source_times.append((time.perf_counter_ns() - started) / STEPS)
        source_state = [
            source.v,
            source.x_nmda,
            source.s_nmda,
            source.ca,
            source.refractory_remaining,
        ]
        retained = SCWBNMDAMagnesiumBlockNeuron()
        started = time.perf_counter_ns()
        for _ in range(STEPS):
            retained.step(5.0)
        sc_times.append((time.perf_counter_ns() - started) / STEPS)
        sc_state = [retained.v, retained.h, retained.n, retained.s_nmda]
    return _summary(source_times, sc_times, source_state, sc_state)


def _parse_lines(lines: list[str]) -> dict[str, object]:
    values: dict[str, list[str]] = {
        key: [] for key in ("source_ns", "sc_ns", "source_state", "sc_state")
    }
    for line in lines:
        for key in values:
            if line.startswith(f"{key}="):
                values[key].append(line.split("=", 1)[1])
    return _summary(
        [float(value) for value in values["source_ns"]],
        [float(value) for value in values["sc_ns"]],
        [float(value) for value in values["source_state"][-1].split(",")],
        [float(value) for value in values["sc_state"][-1].split(",")],
    )


def _rust() -> dict[str, object]:
    lines = subprocess.run(
        [
            "cargo",
            "run",
            "--no-default-features",
            "--manifest-path",
            "engine/Cargo.toml",
            "--example",
            "bench_nmda_rk2",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    ).stdout.splitlines()
    return _parse_lines(lines)


def _go() -> dict[str, object]:
    output = subprocess.run(
        [
            "go",
            "test",
            "./services",
            "-run",
            "^$",
            "-bench",
            "Benchmark(NMDASourceRK2|SCWBNMDAMagnesiumBlock)$",
            "-benchtime",
            f"{STEPS}x",
            "-count",
            str(REPEATS),
        ],
        cwd=ROOT / "src/sc_neurocore/accel/go",
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    ).stdout
    source = [
        float(value)
        for value in re.findall(r"BenchmarkNMDASourceRK2(?:-\d+)?\s+\d+\s+([\d.]+) ns/op", output)
    ]
    sc = [
        float(value)
        for value in re.findall(
            r"BenchmarkSCWBNMDAMagnesiumBlock(?:-\d+)?\s+\d+\s+([\d.]+) ns/op",
            output,
        )
    ]
    return _summary(source, sc, [], [])


def _julia() -> dict[str, object]:
    script = textwrap.dedent(f"""
        include("src/sc_neurocore/accel/julia/neurons/nmda_neuron.jl")
        include("src/sc_neurocore/accel/julia/neurons/sc_wb_nmda_magnesium_block.jl")
        for _ in 1:{REPEATS}
            s=NmdaNeuronAccel.NMDANeuronState(); t=time_ns()
            for _ in 1:{STEPS}; NmdaNeuronAccel.step!(s,0.6); end
            println("source_ns=",(time_ns()-t)/{STEPS})
            println("source_state=",s.v,",",s.x_nmda,",",s.s_nmda,",",s.ca,",",s.refractory_remaining)
            x=SCWBNMDAMagnesiumBlockAccel.SCWBNMDAMagnesiumBlockNeuronState(); t=time_ns()
            for _ in 1:{STEPS}; SCWBNMDAMagnesiumBlockAccel.step!(x,5.0); end
            println("sc_ns=",(time_ns()-t)/{STEPS})
            println("sc_state=",x.v,",",x.h,",",x.n,",",x.s_nmda)
        end
    """)
    lines = subprocess.run(
        ["julia", "--startup-file=no", "-e", script],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    ).stdout.splitlines()
    return _parse_lines(lines)


def _mojo() -> dict[str, object]:
    program = textwrap.dedent(f"""
        from nmda_neuron import NMDANeuron
        from sc_wb_nmda_magnesium_block import SCWBNMDAMagnesiumBlockNeuron
        from std.time import perf_counter
        def main() raises:
            for _ in range({REPEATS}):
                var s=NMDANeuron(); var t=perf_counter()
                for _ in range({STEPS}): _=s.step(0.6)
                print("source_ns=",Float64(perf_counter()-t)*1e9/Float64({STEPS}),sep="")
                print("source_state=",s.v,",",s.x_nmda,",",s.s_nmda,",",s.ca,",",s.refractory_remaining,sep="")
                var x=SCWBNMDAMagnesiumBlockNeuron(); t=perf_counter()
                for _ in range({STEPS}): _=x.step(5.0)
                print("sc_ns=",Float64(perf_counter()-t)*1e9/Float64({STEPS}),sep="")
                print("sc_state=",x.v,",",x.h,",",x.n,",",x.s_nmda,sep="")
    """)
    with tempfile.NamedTemporaryFile("w", suffix=".mojo") as stream:
        stream.write(program)
        stream.flush()
        command = pin_isa(
            [
                "mojo",
                "run",
                "--disable-warnings",
                "-Xlinker",
                "-lm",
                "-I",
                str(ROOT / "src/sc_neurocore/accel/mojo/kernels"),
                stream.name,
            ]
        )
        lines = subprocess.run(
            command, check=True, capture_output=True, text=True, timeout=300
        ).stdout.splitlines()
    return _parse_lines(lines)


def main() -> None:
    backends = {
        name: runner()
        for name, runner in (
            ("python", _python),
            ("rust", _rust),
            ("go", _go),
            ("julia", _julia),
            ("mojo", _mojo),
        )
    }
    payload = {
        "steps": STEPS,
        "repeats": REPEATS,
        "source_current": 0.6,
        "sc_current": 5.0,
        "backends": backends,
        "source_hashes": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCES
        },
        "production_speed_claim": False,
        "hardware_measurement_claimed": False,
        "notes": "Local loaded-host regression; state parity is tested separately; no comparative speed claim.",
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(backends, indent=2))


if __name__ == "__main__":
    main()
