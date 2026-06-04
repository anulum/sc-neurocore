# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Block-floating dense benchmark artefact writer

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
import platform
import statistics
import time

import numpy as np

from sc_neurocore.compiler.quantizer import compile_dense_block_floating


N_INPUTS = 64
N_OUTPUTS = 32
ITERATIONS = 2_000
REPEATS = 7
OUTPUT = Path("benchmarks/results/local_python_2026-06-04_block_floating_dense.json")


def deterministic_inputs() -> tuple[np.ndarray, np.ndarray]:
    weights = np.array(
        [((idx * 23 + 3) % 1025 - 512) / 512.0 for idx in range(N_INPUTS * N_OUTPUTS)],
        dtype=np.float64,
    ).reshape(N_OUTPUTS, N_INPUTS)
    inputs = np.array(
        [((idx * 19 + 5) % 257 - 128) / 256.0 for idx in range(N_INPUTS)],
        dtype=np.float64,
    )
    return weights, inputs


def time_block_floating_forward(compiled, inputs: np.ndarray) -> tuple[float, int]:
    start_ns = time.perf_counter_ns()
    checksum = 0
    for _ in range(ITERATIONS):
        outputs = compiled.forward_accumulator_codes(inputs)
        checksum ^= int(outputs[0])
        checksum ^= int(outputs[-1])
    elapsed_ns = time.perf_counter_ns() - start_ns
    return elapsed_ns / ITERATIONS, checksum


def time_float_dot(weights: np.ndarray, inputs: np.ndarray) -> tuple[float, float]:
    start_ns = time.perf_counter_ns()
    checksum = 0.0
    for _ in range(ITERATIONS):
        outputs = weights @ inputs
        checksum += float(outputs[0]) + float(outputs[-1])
    elapsed_ns = time.perf_counter_ns() - start_ns
    return elapsed_ns / ITERATIONS, checksum


def main() -> int:
    weights, inputs = deterministic_inputs()
    compiled = compile_dense_block_floating(weights, fmt="BFP16E3X32")

    bfp_results = [time_block_floating_forward(compiled, inputs) for _ in range(REPEATS)]
    float_results = [time_float_dot(weights, inputs) for _ in range(REPEATS)]
    bfp_ns = [float(item[0]) for item in bfp_results]
    float_ns = [float(item[0]) for item in float_results]
    reconstructed = compiled.forward_float(inputs)
    reference = weights @ inputs

    report = {
        "benchmark": "block_floating_dense_q16_64x32",
        "language": "Python",
        "timestamp_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "command": "PYTHONPATH=src .venv/bin/python benchmarks/bench_block_floating_dense.py",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "n_inputs": N_INPUTS,
        "n_outputs": N_OUTPUTS,
        "iterations": ITERATIONS,
        "repeats": REPEATS,
        "bfp_median_ns_per_call": statistics.median(bfp_ns),
        "bfp_min_ns_per_call": min(bfp_ns),
        "bfp_max_ns_per_call": max(bfp_ns),
        "float_dot_median_ns_per_call": statistics.median(float_ns),
        "float_dot_min_ns_per_call": min(float_ns),
        "float_dot_max_ns_per_call": max(float_ns),
        "max_abs_error_vs_float_dot": float(np.max(np.abs(reconstructed - reference))),
        "compiled_manifest": compiled.manifest(),
        "bfp_results": [
            {"ns_per_call": result[0], "checksum": result[1]} for result in bfp_results
        ],
        "float_dot_results": [
            {"ns_per_call": result[0], "checksum": result[1]} for result in float_results
        ],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
