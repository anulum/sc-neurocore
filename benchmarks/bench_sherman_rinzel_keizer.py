# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Sherman-Rinzel-Keizer polyglot benchmark

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
import textwrap
import time
from typing import Any

from sc_neurocore.neurons.models.sherman_rinzel_keizer import ShermanRinzelKeizerNeuron

ROOT = Path(__file__).resolve().parents[1]
GO_ROOT = ROOT / "src" / "sc_neurocore" / "accel" / "go"
JULIA_SOURCE = (
    ROOT / "src" / "sc_neurocore" / "accel" / "julia" / "neurons" / "sherman_rinzel_keizer.jl"
)
RUST_SOURCE = (
    ROOT / "src" / "sc_neurocore" / "accel" / "rust" / "safety" / "sherman_rinzel_keizer.rs"
)
RESULT = ROOT / "benchmarks" / "results" / "bench_sherman_rinzel_keizer.json"
STEPS = 80_000
CURRENT = 5.0

BackendResult = dict[str, float | int]


def _parse_backend_csv(output: str) -> BackendResult:
    wall_raw, spikes_raw, v_raw, n_raw, s_raw = output.strip().split(",")
    wall = float(wall_raw)
    return {
        "steps": STEPS,
        "wall_seconds": wall,
        "steps_per_second": STEPS / wall,
        "spikes": int(spikes_raw),
        "v": float(v_raw),
        "n": float(n_raw),
        "s": float(s_raw),
    }


def _bench_python() -> BackendResult:
    neuron = ShermanRinzelKeizerNeuron()
    start = time.perf_counter()
    spikes = 0
    for _ in range(STEPS):
        spikes += neuron.step(CURRENT)
    wall = time.perf_counter() - start
    return {
        "steps": STEPS,
        "wall_seconds": wall,
        "steps_per_second": STEPS / wall,
        "spikes": spikes,
        "v": neuron.v,
        "n": neuron.n,
        "s": neuron.s,
    }


def _bench_go() -> BackendResult:
    source = textwrap.dedent(
        f"""
        package main

        import (
            "fmt"
            "time"

            "github.com/anulum/sc-neurocore/accel/services"
        )

        func main() {{
            neuron := services.NewShermanRinzelKeizerNeuron()
            start := time.Now()
            spikes := 0
            for i := 0; i < {STEPS}; i++ {{
                spikes += neuron.Step({CURRENT})
            }}
            elapsed := time.Since(start).Seconds()
            fmt.Printf("%.17f,%d,%.17f,%.17f,%.17f\\n", elapsed, spikes, neuron.V, neuron.N, neuron.S)
        }}
        """
    )
    with tempfile.TemporaryDirectory(prefix="srk_go_bench_") as tmp:
        harness = Path(tmp) / "main.go"
        harness.write_text(source, encoding="utf-8")
        output = subprocess.check_output(["go", "run", str(harness)], text=True, cwd=GO_ROOT)
    return _parse_backend_csv(output)


def _bench_julia() -> BackendResult:
    julia_script = textwrap.dedent(
        f'''
        include("{JULIA_SOURCE}")
        using .ShermanRinzelKeizerAccel
        function run_benchmark()
            neuron = ShermanRinzelKeizerNeuronState()
            spikes = 0
            start = time()
            for _ in 1:{STEPS}
                spikes += step!(neuron, {CURRENT})
            end
            elapsed = time() - start
            println(string(elapsed, ",", spikes, ",", neuron.v, ",", neuron.n, ",", neuron.s))
        end
        run_benchmark()
        '''
    )
    output = subprocess.check_output(
        ["julia", "--project=.", "-e", julia_script], text=True, cwd=ROOT
    )
    return _parse_backend_csv(output)


def _bench_rust() -> BackendResult:
    with tempfile.TemporaryDirectory(prefix="srk_rust_bench_") as tmp:
        tmp_path = Path(tmp)
        harness = tmp_path / "srk_bench.rs"
        binary = tmp_path / "srk_bench"
        harness.write_text(
            textwrap.dedent(
                f'''
                #[path = "{RUST_SOURCE}"]
                mod srk;
                use std::time::Instant;

                fn main() {{
                    let mut neuron = srk::ShermanRinzelKeizerNeuron::new();
                    let start = Instant::now();
                    let mut spikes: i32 = 0;
                    for _ in 0..{STEPS} {{
                        spikes += neuron.step({CURRENT});
                    }}
                    let elapsed = start.elapsed().as_secs_f64();
                    println!("{{:.17}},{{}},{{:.17}},{{:.17}},{{:.17}}", elapsed, spikes, neuron.v, neuron.n, neuron.s);
                }}
                '''
            ),
            encoding="utf-8",
        )
        subprocess.run(["rustc", "-O", str(harness), "-o", str(binary)], check=True, cwd=ROOT)
        output = subprocess.check_output([str(binary)], text=True, cwd=ROOT)
    return _parse_backend_csv(output)


def _parity_against_python(
    python: BackendResult, backend: BackendResult
) -> dict[str, float | int | str]:
    return {
        "status": "measured",
        "max_abs_delta": max(
            abs(float(python["v"]) - float(backend["v"])),
            abs(float(python["n"]) - float(backend["n"])),
            abs(float(python["s"]) - float(backend["s"])),
        ),
        "spikes_delta": int(python["spikes"]) - int(backend["spikes"]),
    }


def main() -> None:
    backends: dict[str, BackendResult] = {
        "python": _bench_python(),
        "go_service": _bench_go(),
        "julia_mirror": _bench_julia(),
        "rust_safety": _bench_rust(),
    }
    python = backends["python"]
    parity = {
        name: _parity_against_python(python, result)
        for name, result in backends.items()
        if name != "python"
    }
    speedups = {
        name: float(result["steps_per_second"]) / float(python["steps_per_second"])
        for name, result in backends.items()
        if name != "python"
    }
    result: dict[str, Any] = {
        "model": "ShermanRinzelKeizerNeuron",
        "date": "2026-06-01",
        "steps": STEPS,
        "current": CURRENT,
        "backends": backends,
        "speedup_vs_python": speedups,
        "parity_vs_python": parity,
    }
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("Backend            Steps/s    Wall (s)    Speedup    Spikes")
    print("-------------------------------------------------------------")
    for name, backend in backends.items():
        speedup = 1.0 if name == "python" else speedups[name]
        print(
            f"{name:<16} {backend['steps_per_second']:>10.0f} "
            f"{backend['wall_seconds']:>10.6f} {speedup:>9.2f}x {backend['spikes']:>9}"
        )
    print(f"Parity vs Python: {parity}")
    print(f"Results -> {RESULT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
