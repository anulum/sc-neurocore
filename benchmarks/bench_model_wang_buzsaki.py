#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Wang-Buzsaki five-runtime benchmark

"""Measure the source-bound Wang-Buzsaki recurrence across five runtimes."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import tempfile
import textwrap
import time

from sc_neurocore.accel.mojo.isa_baseline import pin_isa
from sc_neurocore.neurons.models import WangBuzsakiNeuron

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "benchmarks/results/bench_wang_buzsaki.json"
STEPS = 20_000
REPEATS = 3
CURRENT = 10.0
BACKENDS = ("python", "rust", "go", "julia", "mojo")
PARITY_ATOL = 1e-8
SOURCES = (
    "benchmarks/bench_model_wang_buzsaki.py",
    "engine/Cargo.toml",
    "engine/examples/bench_wang_buzsaki.rs",
    "engine/src/neurons/biophysical/wang_buzsaki.rs",
    "src/sc_neurocore/neurons/models/wang_buzsaki.py",
    "src/sc_neurocore/accel/go/services/wang_buzsaki.go",
    "src/sc_neurocore/accel/go/services/wang_buzsaki_test.go",
    "src/sc_neurocore/accel/julia/neurons/wang_buzsaki.jl",
    "src/sc_neurocore/accel/mojo/kernels/wang_buzsaki.mojo",
    "src/sc_neurocore/accel/rust/safety/wang_buzsaki.rs",
)


def _summary(
    timings: list[float], spikes: list[int], states: list[list[float]]
) -> dict[str, object]:
    if len(timings) != REPEATS or len(spikes) != REPEATS or len(states) != REPEATS:
        raise RuntimeError("backend did not emit every benchmark repeat")
    if len(set(spikes)) != 1:
        raise RuntimeError(f"backend spike count changed between repeats: {spikes}")
    return {
        "available": True,
        "used": True,
        "median_ns_per_step": statistics.median(timings),
        "results_ns_per_step": timings,
        "spike_count": spikes[-1],
        "final_state": states[-1],
    }


def _parse_lines(lines: list[str]) -> dict[str, object]:
    timings: list[float] = []
    spikes: list[int] = []
    states: list[list[float]] = []
    for line in lines:
        if line.startswith("ns="):
            timings.append(float(line.split("=", 1)[1]))
        elif line.startswith("spikes="):
            spikes.append(int(line.split("=", 1)[1]))
        elif line.startswith("state="):
            states.append([float(value) for value in line.split("=", 1)[1].split(",")])
    return _summary(timings, spikes, states)


def _python() -> dict[str, object]:
    timings: list[float] = []
    spikes: list[int] = []
    states: list[list[float]] = []
    for _ in range(REPEATS):
        neuron = WangBuzsakiNeuron()
        count = 0
        started = time.perf_counter_ns()
        for _ in range(STEPS):
            count += neuron.step(CURRENT)
        timings.append((time.perf_counter_ns() - started) / STEPS)
        spikes.append(count)
        states.append([neuron.v, neuron.h, neuron.n])
    return _summary(timings, spikes, states)


def _rust() -> dict[str, object]:
    completed = subprocess.run(
        [
            "cargo",
            "run",
            "--release",
            "--no-default-features",
            "--manifest-path",
            "engine/Cargo.toml",
            "--example",
            "bench_wang_buzsaki",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    )
    return _parse_lines(completed.stdout.splitlines())


def _go() -> dict[str, object]:
    program = textwrap.dedent(f"""
        package main
        import (
            "fmt"
            "time"
            "github.com/anulum/sc-neurocore/accel/services"
        )
        func main() {{
            for repeat := 0; repeat < {REPEATS}; repeat++ {{
                state := services.NewWangBuzsakiNeuron()
                spikes := 0
                started := time.Now()
                for step := 0; step < {STEPS}; step++ {{
                    event, err := state.Step({CURRENT})
                    if err != nil {{ panic(err) }}
                    spikes += event
                }}
                fmt.Printf("ns=%v\\n", float64(time.Since(started).Nanoseconds())/{STEPS})
                fmt.Printf("spikes=%d\\n", spikes)
                fmt.Printf("state=%.17g,%.17g,%.17g\\n", state.V, state.H, state.N)
            }}
        }}
    """)
    with tempfile.NamedTemporaryFile("w", suffix=".go") as stream:
        stream.write(program)
        stream.flush()
        completed = subprocess.run(
            ["go", "run", stream.name],
            cwd=ROOT / "src/sc_neurocore/accel/go",
            check=True,
            capture_output=True,
            text=True,
            timeout=180,
        )
    return _parse_lines(completed.stdout.splitlines())


def _julia() -> dict[str, object]:
    script = textwrap.dedent(f"""
        include("src/sc_neurocore/accel/julia/neurons/wang_buzsaki.jl")
        for _ in 1:{REPEATS}
            state=WangBuzsakiAccel.WangBuzsakiNeuronState(); spikes=0; started=time_ns()
            for _ in 1:{STEPS}; spikes += WangBuzsakiAccel.step!(state,{CURRENT}); end
            println("ns=",(time_ns()-started)/{STEPS})
            println("spikes=",spikes)
            println("state=",state.v,",",state.h,",",state.n)
        end
    """)
    completed = subprocess.run(
        ["julia", "--startup-file=no", "-e", script],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
    )
    return _parse_lines(completed.stdout.splitlines())


def _mojo() -> dict[str, object]:
    program = textwrap.dedent(f"""
        from wang_buzsaki import WangBuzsakiNeuron
        from std.time import perf_counter
        def main() raises:
            for _ in range({REPEATS}):
                var state=WangBuzsakiNeuron(); var spikes=0; var started=perf_counter()
                for _ in range({STEPS}): spikes += state.step({CURRENT})
                print("ns=",Float64(perf_counter()-started)*1e9/Float64({STEPS}),sep="")
                print("spikes=",spikes,sep="")
                print("state=",state.v,",",state.h,",",state.n,sep="")
    """)
    with tempfile.NamedTemporaryFile("w", suffix=".mojo") as stream:
        stream.write(program)
        stream.flush()
        completed = subprocess.run(
            pin_isa(
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
            ),
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=180,
        )
    return _parse_lines(completed.stdout.splitlines())


def _bind_parity(backends: dict[str, dict[str, object]]) -> None:
    reference = backends["python"]
    reference_spikes = int(reference["spike_count"])
    reference_state = [float(value) for value in reference["final_state"]]  # type: ignore[arg-type]
    for row in backends.values():
        state = [float(value) for value in row["final_state"]]  # type: ignore[arg-type]
        max_diff = max(abs(actual - expected) for actual, expected in zip(state, reference_state))
        row["spike_count_matches_python"] = int(row["spike_count"]) == reference_spikes
        row["parity_max_abs_diff"] = max_diff
        row["final_state_matches_python"] = max_diff <= PARITY_ATOL
        if not row["spike_count_matches_python"] or not row["final_state_matches_python"]:
            raise RuntimeError(f"five-runtime Wang-Buzsaki parity failed: {backends}")


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
    _bind_parity(backends)
    payload = {
        "schema_version": "sc-neurocore.polyglot-benchmark.v1",
        "model": "WangBuzsakiNeuron",
        "steps": STEPS,
        "repeats": REPEATS,
        "current": CURRENT,
        "measured_order": list(BACKENDS),
        "backends": backends,
        "source_hashes": {
            source: hashlib.sha256((ROOT / source).read_bytes()).hexdigest() for source in SOURCES
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "single_cpu_pinned": len(os.sched_getaffinity(0)) == 1,
            "exclusive_cpu_isolation_claimed": False,
        },
        "production_speed_claim": False,
        "hardware_measurement_claimed": False,
        "notes": "Loaded-host regression only; timings are not comparative production claims.",
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(backends, indent=2))


if __name__ == "__main__":
    main()
