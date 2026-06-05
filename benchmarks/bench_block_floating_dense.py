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
from typing import Protocol

import numpy as np

from _benchmark_context import load_average, measurement_context
from sc_neurocore.compiler.quantizer import (
    Q16_16,
    PrecisionEnvelopeReport,
    compile_dense_block_floating,
    quantize_weights,
)


N_INPUTS = 64
N_OUTPUTS = 32
ITERATIONS = 2_000
REPEATS = 7
OUTPUT = Path("benchmarks/results/local_python_2026-06-04_block_floating_dense.json")


class BlockFloatingDenseRunner(Protocol):
    """Compiled block-floating dense object used by the benchmark."""

    def forward_accumulator_codes(self, inputs: np.ndarray) -> np.ndarray:
        """Return saturated Q16.16 output codes."""

    def forward_with_overflow(self, inputs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return saturated Q16.16 output codes and per-output overflow mask."""

    def precision_envelope_report(self, inputs: np.ndarray) -> PrecisionEnvelopeReport:
        """Return conservative per-output precision envelope telemetry."""


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


def time_block_floating_forward(
    compiled: BlockFloatingDenseRunner,
    inputs: np.ndarray,
) -> tuple[float, int]:
    start_ns = time.perf_counter_ns()
    checksum = 0
    for _ in range(ITERATIONS):
        outputs = compiled.forward_accumulator_codes(inputs)
        checksum ^= int(outputs[0])
        checksum ^= int(outputs[-1])
    elapsed_ns = time.perf_counter_ns() - start_ns
    return elapsed_ns / ITERATIONS, checksum


def time_block_floating_forward_with_overflow(
    compiled: BlockFloatingDenseRunner,
    inputs: np.ndarray,
) -> tuple[float, int, int]:
    start_ns = time.perf_counter_ns()
    checksum = 0
    overflow_count = 0
    for _ in range(ITERATIONS):
        outputs, overflow = compiled.forward_with_overflow(inputs)
        checksum ^= int(outputs[0])
        checksum ^= int(outputs[-1])
        overflow_count = int(np.count_nonzero(overflow))
    elapsed_ns = time.perf_counter_ns() - start_ns
    return elapsed_ns / ITERATIONS, checksum, overflow_count


def time_float_dot(weights: np.ndarray, inputs: np.ndarray) -> tuple[float, float]:
    start_ns = time.perf_counter_ns()
    checksum = 0.0
    for _ in range(ITERATIONS):
        outputs = weights @ inputs
        checksum += float(outputs[0]) + float(outputs[-1])
    elapsed_ns = time.perf_counter_ns() - start_ns
    return elapsed_ns / ITERATIONS, checksum


def exponent_edge_sweep_report() -> dict[str, object]:
    weights = np.array(
        [
            [0.125, -0.25, 1_000_000.0, -1_000_000.0],
            [-0.375, 0.5, -1_000_000.0, 1_000_000.0],
        ],
        dtype=np.float64,
    )
    inputs = np.array([0.5, -0.25, 1 / 65536.0, -1 / 65536.0], dtype=np.float64)
    compiled = compile_dense_block_floating(weights, fmt="BFP16E3X2")
    reconstructed = compiled.forward_float(inputs)
    q_inputs = quantize_weights(inputs, fmt=Q16_16).astype(np.float64) / Q16_16.scale
    reference = compiled.reconstructed_weights @ q_inputs
    codes, overflow = compiled.forward_with_overflow(inputs)
    envelope = compiled.precision_envelope_report(inputs)

    saturating = compile_dense_block_floating(
        np.array([[1_000_000.0, 1_000_000.0]], dtype=np.float64),
        fmt="BFP16E3X2",
    )
    saturating_codes, saturating_overflow = saturating.forward_with_overflow(
        np.array([32767.0, 32767.0], dtype=np.float64)
    )
    saturating_envelope = saturating.precision_envelope_report(
        np.array([32767.0, 32767.0], dtype=np.float64)
    )

    return {
        "format": "BFP16E3X2",
        "safe_exponent_codes": compiled.exponents.astype(int).tolist(),
        "safe_output_codes_q1616": codes.astype(int).tolist(),
        "safe_overflow_count": int(np.count_nonzero(overflow)),
        "safe_underflow_count": envelope.underflow_count,
        "safe_max_abs_error_vs_reconstructed_reference": float(
            np.max(np.abs(reconstructed - reference))
        ),
        "safe_max_abs_bound_q1616": envelope.max_abs_bound_code,
        "safe_min_headroom_q1616": envelope.min_headroom_code,
        "safe_conservative_overflow_free": envelope.conservative_overflow_free,
        "max_exponent_saturating_codes_q1616": saturating_codes.astype(int).tolist(),
        "max_exponent_saturating_exponent_codes": saturating.exponents.astype(int).tolist(),
        "max_exponent_saturating_overflow_count": int(np.count_nonzero(saturating_overflow)),
        "max_exponent_saturating_underflow_count": saturating_envelope.underflow_count,
        "max_exponent_saturating_conservative_overflow_free": (
            saturating_envelope.conservative_overflow_free
        ),
        "max_exponent_saturating_max_abs_bound_q1616": (
            saturating_envelope.max_abs_bound_code
        ),
    }


def main() -> int:
    load_average_before = load_average()
    weights, inputs = deterministic_inputs()
    compiled = compile_dense_block_floating(weights, fmt="BFP16E3X32")

    bfp_results = [time_block_floating_forward(compiled, inputs) for _ in range(REPEATS)]
    overflow_results = [
        time_block_floating_forward_with_overflow(compiled, inputs) for _ in range(REPEATS)
    ]
    float_results = [time_float_dot(weights, inputs) for _ in range(REPEATS)]
    bfp_ns = [float(item[0]) for item in bfp_results]
    overflow_ns = [float(item[0]) for item in overflow_results]
    float_ns = [float(item[0]) for item in float_results]
    reconstructed = compiled.forward_float(inputs)
    reference = weights @ inputs
    _, safe_overflow = compiled.forward_with_overflow(inputs)
    safe_envelope = compiled.precision_envelope_report(inputs)
    overflow_probe = compile_dense_block_floating(
        np.full((N_OUTPUTS, N_INPUTS), 8192.0, dtype=np.float64),
        fmt="BFP16E3X32",
    )
    _, probe_overflow = overflow_probe.forward_with_overflow(
        np.full(N_INPUTS, 32767.0, dtype=np.float64)
    )
    probe_envelope = overflow_probe.precision_envelope_report(
        np.full(N_INPUTS, 32767.0, dtype=np.float64)
    )

    report = {
        "benchmark": "block_floating_dense_q16_64x32",
        "language": "Python",
        "timestamp_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "command": (
            "taskset -c 8-9 env PYTHONPATH=src "
            ".venv/bin/python benchmarks/bench_block_floating_dense.py"
        ),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "measurement_context": measurement_context(load_average_before),
        "n_inputs": N_INPUTS,
        "n_outputs": N_OUTPUTS,
        "iterations": ITERATIONS,
        "repeats": REPEATS,
        "bfp_median_ns_per_call": statistics.median(bfp_ns),
        "bfp_min_ns_per_call": min(bfp_ns),
        "bfp_max_ns_per_call": max(bfp_ns),
        "forward_with_overflow_median_ns_per_call": statistics.median(overflow_ns),
        "forward_with_overflow_min_ns_per_call": min(overflow_ns),
        "forward_with_overflow_max_ns_per_call": max(overflow_ns),
        "float_dot_median_ns_per_call": statistics.median(float_ns),
        "float_dot_min_ns_per_call": min(float_ns),
        "float_dot_max_ns_per_call": max(float_ns),
        "max_abs_error_vs_float_dot": float(np.max(np.abs(reconstructed - reference))),
        "safe_overflow_count": int(np.count_nonzero(safe_overflow)),
        "safe_max_abs_bound_code": safe_envelope.max_abs_bound_code,
        "safe_conservative_overflow_free": safe_envelope.conservative_overflow_free,
        "safe_min_headroom_code": safe_envelope.min_headroom_code,
        "mantissa_checksum": int(np.sum(compiled.mantissas.astype(np.int64))),
        "exponent_checksum": int(np.sum(compiled.exponents.astype(np.int64))),
        "block_exponent_count": compiled.manifest()["block_exponent_count"],
        "exponent_code_min": int(np.min(compiled.exponents)),
        "exponent_code_max": int(np.max(compiled.exponents)),
        "saturating_probe_overflow_count": int(np.count_nonzero(probe_overflow)),
        "saturating_probe_max_abs_bound_code": probe_envelope.max_abs_bound_code,
        "saturating_probe_conservative_overflow_free": probe_envelope.conservative_overflow_free,
        "exponent_edge_sweep": exponent_edge_sweep_report(),
        "compiled_manifest": compiled.manifest(),
        "bfp_results": [
            {"ns_per_call": result[0], "checksum": result[1]} for result in bfp_results
        ],
        "forward_with_overflow_results": [
            {"ns_per_call": result[0], "checksum": result[1], "overflow_count": result[2]}
            for result in overflow_results
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
