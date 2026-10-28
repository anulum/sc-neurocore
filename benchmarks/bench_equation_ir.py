# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source-bound native equation compilation benchmark

"""Measure complete public equation compilation, replay and both hardware emitters."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
import time
from collections.abc import Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks._benchmark_context import measurement_context
from sc_neurocore.compiler import compile_to_ir, recompile_ir
from sc_neurocore.neurons.equation_builder import EquationNeuron


@dataclass(frozen=True)
class Case:
    """An actual model and its product-register request.

    Attributes
    ----------
    name : str
        Workload and hardware module name.
    model : EquationNeuron
        Initial model, preserved between compilation samples.
    stages : int
        Maintained global product-register selection.
    """

    name: str
    model: EquationNeuron
    stages: int = 0


def _cases() -> tuple[Case, ...]:
    return (
        Case("bench_rsqrt", EquationNeuron({"v": "rsqrt(I)"}, method="map", dt=1.0)),
        Case("bench_hypot", EquationNeuron({"v": "hypot(I,1)"}, method="map", dt=1.0)),
        Case("bench_round", EquationNeuron({"v": "round(I)"}, method="map", dt=1.0)),
        Case(
            "bench_heaviside",
            EquationNeuron(
                {"v": "heaviside(I, z)", "z": "-z"},
                state={"v": 0.0, "z": 0.25},
                method="map",
                dt=1.0,
            ),
        ),
        Case(
            "bench_euler", EquationNeuron({"v": "(I - v) / tau"}, parameters={"tau": 8.0}, dt=0.5)
        ),
        Case(
            "bench_rk4",
            EquationNeuron(
                {"v": "gain * (I - v + w)", "w": "(v - w) / tau"},
                parameters={"gain": 0.25},
                constants={"tau": 8.0},
                state={"v": -0.5, "w": 0.75},
                method="rk4",
                dt=0.5,
                threshold="v > 2",
                reset={"v": "w", "w": "v_prev"},
                spike_value="v - w_prev",
            ),
            1,
        ),
        Case(
            "bench_random",
            EquationNeuron(
                {"v": "rand() * gain + I"},
                parameters={"gain": 0.5},
                method="map",
                dt=1.0,
                state={"v": -0.5},
                threshold="v > 1",
                reset={"v": "0"},
                rng_seed=(1 << 63) + 11,
                random_stream=(1 << 31) + 7,
            ),
        ),
        Case(
            "bench_macro",
            EquationNeuron(
                {"v": "I", "w": "w + I / 2"},
                state={"v": 0.25, "w": -0.5},
                method="map",
                dt=1.0,
                threshold="v > v_prev",
                spike_value="v - v_prev",
                detection="crossing",
                substeps=3,
            ),
        ),
    )


def _compile(case: Case) -> tuple[str, int, int, int, str]:
    result = compile_to_ir(
        case.model,
        case.name,
        rounding="bankers",
        function_lowering="v2",
        pipeline_stages=case.stages,
    )
    text = result.graph.to_text()
    if recompile_ir(text).graph.to_text() != text:
        raise RuntimeError(f"source replay changed graph: {case.name}")
    hardware = result.graph.emit_sv() + "\0" + result.graph.emit_mlir()
    source = result.graph.source_payload
    if source is None:
        raise RuntimeError(f"compiled graph has no model source: {case.name}")
    return (
        hashlib.sha256((text + "\0" + hardware).encode()).hexdigest(),
        len(result.graph),
        result.clocks_per_timestep,
        result.pipeline_registers,
        source,
    )


def run_benchmark(samples: int, warmup: int, command: str) -> dict[str, object]:
    """Measure deterministic full compiler transactions on the observed host.

    Parameters
    ----------
    samples : int
        Positive number of measured calls per model.
    warmup : int
        Nonnegative unmeasured calls per model.
    command : str
        Reproduction command with repository-relative paths.

    Returns
    -------
    dict of str to object
        Raw timings, output checksums, source/binary hashes and actual hardware
        context. Measurements describe the Python frontend with its Rust backend;
        they are not independent language implementations or reserved-core results.

    Raises
    ------
    ValueError
        Sampling counts are out of range.
    RuntimeError
        Actual graph or hardware output changes during measurement.
    """
    if samples < 1 or warmup < 0:
        raise ValueError("samples must be positive and warmup nonnegative")
    cases = _cases()
    expected = {case.name: _compile(case) for case in cases}
    for case in cases:
        for _ in range(warmup):
            if _compile(case) != expected[case.name]:
                raise RuntimeError(f"warmup changed output: {case.name}")
    before = list(os.getloadavg())
    timings: dict[str, list[int]] = {case.name: [] for case in cases}
    for sample in range(samples):
        offset = sample % len(cases)
        for case in (*cases[offset:], *cases[:offset]):
            started = time.perf_counter_ns()
            observed = _compile(case)
            timings[case.name].append(time.perf_counter_ns() - started)
            if observed != expected[case.name]:
                raise RuntimeError(f"sample changed output: {case.name}")
    paths = [
        *ROOT.glob("src/sc_neurocore/compiler/**/*.py"),
        *ROOT.glob("src/sc_neurocore/neurons/equation*.py"),
        ROOT / "src/sc_neurocore/neurons/_units.py",
        ROOT / "src/sc_neurocore/neurons/expression_derivative.py",
        *ROOT.glob("engine/src/ir/**/*.rs"),
        ROOT / "bridge/sc_neurocore_engine/ir.py",
        ROOT / "bridge/sc_neurocore_engine/_ir_native.py",
        ROOT / "Cargo.lock",
        ROOT / "engine/Cargo.toml",
        Path(__file__),
        ROOT / "benchmarks/_benchmark_context.py",
    ]
    native = importlib.import_module("sc_neurocore_engine.sc_neurocore_engine")
    if native.__file__ is None:
        raise RuntimeError("native engine has no measured binary origin")
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "source_base_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source_state": "base plus source-hash-bound working overlay",
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
        },
        "native_sha256": hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("sc-neurocore", "sc-neurocore-engine", "numpy", "sympy", "pint")
        },
        "measurement_context": measurement_context(before),
        "load_average_after": list(os.getloadavg()),
        "samples_per_case": samples,
        "warmup_per_case": warmup,
        "timed_region": "compile_to_ir, source replay, both native emitters and output checksum",
        "implementation_scope": "Python model frontend with maintained Rust graph backend",
        "cross_language_comparison": False,
        "isolation": "shared loaded host; no exclusive core reservation",
        "performance_promotion_claimed": False,
        "hardware_measurement_claimed": False,
        "cases": {
            case.name: {
                "samples_ns": timings[case.name],
                "median_ms": statistics.median(timings[case.name]) / 1e6,
                "output_sha256": expected[case.name][0],
                "operations": expected[case.name][1],
                "clocks_per_timestep": expected[case.name][2],
                "pipeline_registers": expected[case.name][3],
                "model_source": json.loads(expected[case.name][4]),
            }
            for case in cases
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Write a new source-bound compiler timing capture.

    Parameters
    ----------
    argv : sequence of str or None, optional
        CLI sampling options and a new output path; None reads process arguments.

    Returns
    -------
    int
        Zero after a successful capture. An existing output file is not replaced.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=11)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--json", type=Path, required=True)
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(arguments)
    payload = run_benchmark(
        args.samples,
        args.warmup,
        shlex.join(("python", "benchmarks/bench_equation_ir.py", *arguments)),
    )
    args.json.parent.mkdir(parents=True, exist_ok=True)
    with args.json.open("x") as output:
        json.dump(payload, output, indent=2, sort_keys=True)
        output.write("\n")
    print("Wrote", args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
