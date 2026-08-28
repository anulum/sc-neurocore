#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Larter-Breakspear dual-identity five-runtime benchmark

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import re
import statistics
import subprocess
import tempfile
import textwrap
import time

from sc_neurocore.accel.mojo.isa_baseline import pin_isa
from sc_neurocore.neurons.models import (
    LarterBreakspearNeuron,
    SCDecoupledAdaptationIonMassNeuron,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "benchmarks/results/bench_larter_breakspear.json"
STEPS = 20_000
REPEATS = 3
SOURCES = [
    "benchmarks/bench_model_larter_breakspear.py",
    "engine/examples/bench_larter_breakspear_rk4.rs",
    "engine/src/neurons/special/larter_breakspear_neural_mass.rs",
    "engine/src/neurons/special/sc_decoupled_adaptation_ion_mass.rs",
    "src/sc_neurocore/neurons/models/larter_breakspear.py",
    "src/sc_neurocore/neurons/models/sc_decoupled_adaptation_ion_mass.py",
    "src/sc_neurocore/accel/go/services/larter_breakspear.go",
    "src/sc_neurocore/accel/go/services/sc_decoupled_adaptation_ion_mass.go",
    "src/sc_neurocore/accel/julia/neurons/larter_breakspear.jl",
    "src/sc_neurocore/accel/julia/neurons/sc_decoupled_adaptation_ion_mass.jl",
    "src/sc_neurocore/accel/mojo/kernels/larter_breakspear.mojo",
    "src/sc_neurocore/accel/mojo/kernels/sc_decoupled_adaptation_ion_mass.mojo",
    "src/sc_neurocore/accel/rust/safety/larter_breakspear.rs",
    "src/sc_neurocore/accel/rust/safety/sc_decoupled_adaptation_ion_mass.rs",
    "src/sc_neurocore/neurons/model_descriptors/LarterBreakspearNeuron.toml",
    "src/sc_neurocore/neurons/model_descriptors/SCDecoupledAdaptationIonMassNeuron.toml",
    "src/sc_neurocore/neurons/model_schemas/larter_breakspear.json",
    "src/sc_neurocore/neurons/model_schemas/larter_breakspear.toml",
    "src/sc_neurocore/neurons/model_schemas/sc_decoupled_adaptation_ion_mass.json",
    "src/sc_neurocore/neurons/model_schemas/sc_decoupled_adaptation_ion_mass.toml",
    "src/sc_neurocore/neurons/reference_receipts/larter_breakspear_2003.json",
    "src/sc_neurocore/neurons/reference_receipts/sc_decoupled_adaptation_ion_mass.json",
]


def _source_hashes() -> dict[str, object]:
    """Return flat digests plus suffix aliases consumed by the evidence gate."""
    hashes: dict[str, object] = {}
    for relative in SOURCES:
        digest = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        hashes[relative] = digest
        stem, suffix = relative.rsplit(".", 1)
        aliases = hashes.setdefault(stem, {})
        if not isinstance(aliases, dict):
            raise RuntimeError(f"source-hash alias collision at {stem}")
        aliases[suffix] = digest
    return hashes


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
        source = LarterBreakspearNeuron()
        started = time.perf_counter_ns()
        for _ in range(STEPS):
            source.step(0.0)
        source_times.append((time.perf_counter_ns() - started) / STEPS)
        source_state = [source.v, source.w, source.z]
        retained = SCDecoupledAdaptationIonMassNeuron()
        started = time.perf_counter_ns()
        for _ in range(STEPS):
            retained.step(0.0)
        sc_times.append((time.perf_counter_ns() - started) / STEPS)
        sc_state = [retained.v, retained.w, retained.z]
    return _summary(source_times, sc_times, source_state, sc_state)


def _rust() -> dict[str, object]:
    output = subprocess.run(
        [
            "cargo",
            "run",
            "--no-default-features",
            "--manifest-path",
            "engine/Cargo.toml",
            "--example",
            "bench_larter_breakspear_rk4",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    ).stdout.splitlines()
    return _parse_lines(output)


def _parse_lines(lines: list[str]) -> dict[str, object]:
    values: dict[str, list[str]] = {
        key: [] for key in ("source_ns", "sc_ns", "source_state", "sc_state")
    }
    for line in lines:
        for key in values:
            if line.startswith(f"{key}="):
                values[key].append(line.split("=", 1)[1])
    return _summary(
        [float(x) for x in values["source_ns"]],
        [float(x) for x in values["sc_ns"]],
        [float(x) for x in values["source_state"][-1].split(",")],
        [float(x) for x in values["sc_state"][-1].split(",")],
    )


def _go() -> dict[str, object]:
    output = subprocess.run(
        [
            "go",
            "test",
            "./services",
            "-run",
            "^$",
            "-bench",
            "Benchmark(LarterBreakspear|SCDecoupledAdaptationIonMass)RK4$",
            "-benchtime",
            f"{STEPS}x",
            "-count",
            str(REPEATS),
        ],
        cwd=ROOT / "src/sc_neurocore/accel/go",
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    source = [
        float(x)
        for x in re.findall(r"BenchmarkLarterBreakspearRK4(?:-\d+)?\s+\d+\s+([\d.]+) ns/op", output)
    ]
    sc = [
        float(x)
        for x in re.findall(
            r"BenchmarkSCDecoupledAdaptationIonMassRK4(?:-\d+)?\s+\d+\s+([\d.]+) ns/op", output
        )
    ]
    return _summary(source, sc, [], [])


def _julia() -> dict[str, object]:
    script = textwrap.dedent(f"""
        include("src/sc_neurocore/accel/julia/neurons/larter_breakspear.jl")
        include("src/sc_neurocore/accel/julia/neurons/sc_decoupled_adaptation_ion_mass.jl")
        for _ in 1:{REPEATS}
            s=LarterBreakspearAccel.LarterBreakspearNeuronState(); t=time_ns()
            for _ in 1:{STEPS}; LarterBreakspearAccel.step!(s); end
            println("source_ns=",(time_ns()-t)/{STEPS}); println("source_state=",s.v,",",s.w,",",s.z)
            x=SCDecoupledAdaptationIonMassAccel.SCDecoupledAdaptationIonMassNeuronState(); t=time_ns()
            for _ in 1:{STEPS}; SCDecoupledAdaptationIonMassAccel.step!(x); end
            println("sc_ns=",(time_ns()-t)/{STEPS}); println("sc_state=",x.v,",",x.w,",",x.z)
        end
    """)
    lines = subprocess.run(
        ["julia", "--startup-file=no", "-e", script],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
    ).stdout.splitlines()
    return _parse_lines(lines)


def _mojo() -> dict[str, object]:
    program = textwrap.dedent(f"""
        from larter_breakspear import LarterBreakspear
        from sc_decoupled_adaptation_ion_mass import SCDecoupledAdaptationIonMass
        from std.time import perf_counter
        def main() raises:
            for _ in range({REPEATS}):
                var s=LarterBreakspear(); var t=perf_counter()
                _=s.simulate({STEPS},0.0)
                print("source_ns=",Float64(perf_counter()-t)*1e9/Float64({STEPS}),sep="")
                print("source_state=",s.v,",",s.w,",",s.z,sep="")
                var x=SCDecoupledAdaptationIonMass(); t=perf_counter()
                for _ in range({STEPS}): _=x.step(0.0)
                print("sc_ns=",Float64(perf_counter()-t)*1e9/Float64({STEPS}),sep="")
                print("sc_state=",x.v,",",x.w,",",x.z,sep="")
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
            command, check=True, capture_output=True, text=True, timeout=180
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
        "schema_version": "sc-neurocore.polyglot-benchmark.v1",
        "benchmark": "Larter-Breakspear source and retained SC dual-identity RK4",
        "models": ["LarterBreakspearNeuron", "SCDecoupledAdaptationIonMassNeuron"],
        "evidence_class": "local_regression_non_isolated",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "steps": STEPS,
        "repeats": REPEATS,
        "coupling": 0.0,
        "backends": backends,
        "source_hashes": _source_hashes(),
        "production_speed_claim": False,
        "hardware_measurement_claimed": False,
        "notes": "Local loaded-host regression; continuous state parity is tested separately; no comparative speed claim.",
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(backends, indent=2))


if __name__ == "__main__":
    main()
