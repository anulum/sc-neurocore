#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Balanced Resonate-and-Fire Benchmark

"""Reproducible scalar and population benchmark for BRF neurons.

The benchmark records equation-level behaviour and throughput. It does not
claim parity with the ICML 2024 BRF-RSNN task results; those require the full
training recipe and dataset pipeline.
"""

from __future__ import annotations

import json
import platform
import re
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from sc_neurocore.neurons.models.balanced_resonate_and_fire import (
    BalancedResonateAndFireNeuron,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _drive_trace(
    n_steps: int,
    current: float,
    omega: float,
    b_offset: float,
) -> dict[str, float | str]:
    neuron = BalancedResonateAndFireNeuron(omega=omega, b_offset=b_offset)
    t0 = time.perf_counter()
    spikes = 0
    for _ in range(n_steps):
        spikes += neuron.step(current)
    elapsed = time.perf_counter() - t0
    return {
        "backend": "python",
        "status": "executed",
        "n_steps": n_steps,
        "current": current,
        "omega": omega,
        "b_offset": b_offset,
        "elapsed_seconds": elapsed,
        "step_ns": elapsed / n_steps * 1e9,
        "spikes": spikes,
        "final_x": neuron.x,
        "final_y": neuron.y,
        "final_q": neuron.q,
    }


def _population_loop(n_neurons: int, n_steps: int) -> dict[str, float]:
    neurons = [
        BalancedResonateAndFireNeuron(omega=5.0 + 0.05 * idx, b_offset=1.0 + 0.001 * idx)
        for idx in range(n_neurons)
    ]
    currents = np.linspace(1.0, 20.0, n_neurons, dtype=np.float64)
    t0 = time.perf_counter()
    spikes = 0
    for _ in range(n_steps):
        for neuron, current in zip(neurons, currents):
            spikes += neuron.step(float(current))
    elapsed = time.perf_counter() - t0
    updates = n_neurons * n_steps
    return {
        "n_neurons": n_neurons,
        "n_steps": n_steps,
        "updates": updates,
        "elapsed_seconds": elapsed,
        "update_ns": elapsed / updates * 1e9,
        "updates_per_second": updates / elapsed,
        "spikes": spikes,
    }


def _rust_engine_loop(
    n_steps: int,
    current: float,
    omega: float,
    b_offset: float,
) -> dict[str, float | str | dict[str, float]]:
    try:
        import sc_neurocore_engine as engine
    except ImportError as exc:
        return {"backend": "rust_pyo3", "status": f"not importable: {exc}"}

    neuron_cls = getattr(engine, "BalancedResonateAndFireNeuron", None)
    if neuron_cls is None:
        return {
            "backend": "rust_pyo3",
            "status": "not available in installed sc_neurocore_engine build",
        }

    neuron = neuron_cls()
    t0 = time.perf_counter()
    spikes = 0
    for _ in range(n_steps):
        spikes += int(neuron.step(current))
    elapsed = time.perf_counter() - t0
    return {
        "backend": "rust_pyo3",
        "status": "executed",
        "n_steps": n_steps,
        "current": current,
        "omega": omega,
        "b_offset": b_offset,
        "elapsed_seconds": elapsed,
        "step_ns": elapsed / n_steps * 1e9,
        "spikes": spikes,
        "state": neuron.get_state(),
    }


def _go_benchmark() -> dict[str, float | str]:
    if shutil.which("go") is None:
        return {"backend": "go", "status": "go executable not found"}
    command = ["go", "test", "./services", "-run", "^$", "-bench", "^BenchmarkBalancedRFStep$"]
    try:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT / "src/sc_neurocore/accel/go",
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        return {"backend": "go", "status": "timeout"}
    if completed.returncode != 0:
        return {"backend": "go", "status": completed.stderr.strip() or completed.stdout.strip()}
    match = re.search(r"BenchmarkBalancedRFStep-\d+\s+\d+\s+([0-9.]+)\s+ns/op", completed.stdout)
    if not match:
        return {"backend": "go", "status": "benchmark output did not include ns/op"}
    return {
        "backend": "go",
        "status": "executed",
        "step_ns": float(match.group(1)),
        "command": " ".join(command),
    }


def _julia_benchmark() -> dict[str, float | str]:
    julia = shutil.which("julia") or "/home/anulum/.juliaup/bin/julia"
    if not Path(julia).exists():
        return {"backend": "julia", "status": "julia executable not found"}
    code = """
include("src/sc_neurocore/accel/julia/neurons/balanced_resonate_and_fire.jl")
using .BalancedResonateAndFireAccel
let
    n_steps = 200000
    s = BalancedResonateAndFireNeuronState()
    spikes = 0
    t0 = time_ns()
    for _ in 1:n_steps
        spikes += step!(s, 2.0)
    end
    elapsed = (time_ns() - t0) / 1e9
    println("backend julia")
    println("status executed")
    println("n_steps ", n_steps)
    println("current 2.0")
    println("omega 10.0")
    println("b_offset 1.0")
    println("elapsed_seconds ", elapsed)
    println("step_ns ", elapsed / n_steps * 1e9)
    println("spikes ", spikes)
    println("final_x ", s.x)
    println("final_y ", s.y)
    println("final_q ", s.q)
end
"""
    return _run_key_value_benchmark([julia, "-e", code], backend="julia")


def _mojo_benchmark() -> dict[str, float | str]:
    mojo = shutil.which("mojo") or "/home/anulum/.pixi/bin/mojo"
    if not Path(mojo).exists():
        return {"backend": "mojo", "status": "mojo executable not found"}
    return _run_key_value_benchmark(
        [mojo, "run", "src/sc_neurocore/accel/mojo/kernels/balanced_resonate_and_fire_bench.mojo"],
        backend="mojo",
    )


def _run_key_value_benchmark(command: list[str], *, backend: str) -> dict[str, float | str]:
    try:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        return {"backend": backend, "status": "timeout"}
    if completed.returncode != 0:
        return {"backend": backend, "status": completed.stderr.strip() or completed.stdout.strip()}
    result: dict[str, float | str] = {"backend": backend}
    for line in completed.stdout.splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            continue
        key, value = parts
        if key in {"backend", "status"}:
            result[key] = value
        else:
            try:
                result[key] = float(value)
            except ValueError:
                result[key] = value
    result.setdefault("status", "executed")
    return result


def _side_by_side_scalar_row(
    *,
    workload: str,
    python_run: dict[str, float | str],
    rust_run: dict[str, float | str | dict[str, float]],
) -> dict[str, object]:
    row: dict[str, object] = {
        "workload": workload,
        "n_steps": python_run["n_steps"],
        "current": python_run["current"],
        "omega": python_run["omega"],
        "b_offset": python_run["b_offset"],
        "python": {
            "status": python_run["status"],
            "elapsed_seconds": python_run["elapsed_seconds"],
            "step_ns": python_run["step_ns"],
            "spikes": python_run["spikes"],
            "final_state": {
                "x": python_run["final_x"],
                "y": python_run["final_y"],
                "q": python_run["final_q"],
            },
        },
        "rust_pyo3": {
            "status": rust_run["status"],
        },
    }
    if rust_run.get("status") == "executed":
        rust_state = rust_run["state"]
        assert isinstance(rust_state, dict)
        row["rust_pyo3"] = {
            "status": rust_run["status"],
            "elapsed_seconds": rust_run["elapsed_seconds"],
            "step_ns": rust_run["step_ns"],
            "spikes": rust_run["spikes"],
            "final_state": rust_state,
        }
        row["parity"] = {
            "spikes_match": python_run["spikes"] == rust_run["spikes"],
            "abs_x_delta": abs(float(python_run["final_x"]) - float(rust_state["x"])),
            "abs_y_delta": abs(float(python_run["final_y"]) - float(rust_state["y"])),
            "abs_q_delta": abs(float(python_run["final_q"]) - float(rust_state["q"])),
        }
        row["speedup"] = {
            "rust_vs_python_step_ns": float(python_run["step_ns"]) / float(rust_run["step_ns"])
        }
    return row


def main() -> None:
    out_path = Path(__file__).parent / "results" / "bench_balanced_resonate_and_fire.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scalar_i2 = _drive_trace(200_000, current=2.0, omega=10.0, b_offset=1.0)
    scalar_i20 = _drive_trace(200_000, current=20.0, omega=20.0, b_offset=2.0)
    rust_scalar_i2 = _rust_engine_loop(200_000, current=2.0, omega=10.0, b_offset=1.0)
    go_scalar_i2 = _go_benchmark()
    julia_scalar_i2 = _julia_benchmark()
    mojo_scalar_i2 = _mojo_benchmark()
    comparison = [
        _side_by_side_scalar_row(
            workload="scalar_200k_i2_omega10",
            python_run=scalar_i2,
            rust_run=rust_scalar_i2,
        ),
        {
            "workload": "scalar_200k_i2_omega10",
            "n_steps": 200_000,
            "current": 2.0,
            "omega": 10.0,
            "b_offset": 1.0,
            "python_step_ns": scalar_i2["step_ns"],
            "rust_pyo3_step_ns": rust_scalar_i2.get("step_ns"),
            "go_step_ns": go_scalar_i2.get("step_ns"),
            "julia_step_ns": julia_scalar_i2.get("step_ns"),
            "mojo_step_ns": mojo_scalar_i2.get("step_ns"),
            "backend_status": {
                "python": scalar_i2["status"],
                "rust_pyo3": rust_scalar_i2["status"],
                "go": go_scalar_i2["status"],
                "julia": julia_scalar_i2["status"],
                "mojo": mojo_scalar_i2["status"],
            },
        },
    ]
    runs = [
        scalar_i2,
        scalar_i20,
        _population_loop(n_neurons=256, n_steps=2_000),
        rust_scalar_i2,
        go_scalar_i2,
        julia_scalar_i2,
        mojo_scalar_i2,
    ]
    payload = {
        "schema_version": 1,
        "module": "sc_neurocore.neurons.models.balanced_resonate_and_fire",
        "publication_reference": (
            "Higuchi, Kairat, Bohte, and Otte (2024), Balanced "
            "Resonate-and-Fire Neurons, ICML/PMLR 235."
        ),
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "backends": {
            "python": "executed",
            "rust": "engine implementation added; PyO3 benchmark executes when rebuilt extension exposes it",
            "go": "service mirror implemented",
            "julia": "service mirror implemented",
            "mojo": "kernel mirror implemented",
        },
        "comparison": comparison,
        "runs": runs,
    }
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
