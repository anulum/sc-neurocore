# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mixed-precision dense benchmark artefact writer

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import platform
import statistics
import time
from typing import Protocol

import numpy as np

from _benchmark_context import load_average, measurement_context
from sc_neurocore.compiler.quantizer import (
    PrecisionEnvelopeReport,
    QFormatMixed,
    compile_dense_mixed_precision,
)


N_INPUTS = 64
N_OUTPUTS = 32
ITERATIONS = 2_000
REPEATS = 7
OUTPUT = Path("benchmarks/results/local_python_2026-06-04_mixed_dense.json")


class MixedDenseRunner(Protocol):
    """Compiled mixed dense object used by the benchmark."""

    def forward_accumulator_codes(self, inputs: np.ndarray) -> np.ndarray:
        """Return saturated Q16.16 output codes."""

    def forward_with_overflow(self, inputs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return saturated Q16.16 output codes and per-output overflow mask."""

    def precision_envelope_report(self, inputs: np.ndarray) -> PrecisionEnvelopeReport:
        """Return conservative per-output precision envelope telemetry."""


def deterministic_inputs() -> tuple[np.ndarray, np.ndarray]:
    weights = np.array(
        [((idx * 17 + 11) % 513 - 256) / 256.0 for idx in range(N_INPUTS * N_OUTPUTS)],
        dtype=np.float64,
    ).reshape(N_OUTPUTS, N_INPUTS)
    inputs = np.array(
        [((idx * 19 + 5) % 257 - 128) / 256.0 for idx in range(N_INPUTS)],
        dtype=np.float64,
    )
    return weights, inputs


def time_mixed_forward(compiled: MixedDenseRunner, inputs: np.ndarray) -> tuple[float, int]:
    start_ns = time.perf_counter_ns()
    checksum = 0
    for _ in range(ITERATIONS):
        outputs = compiled.forward_accumulator_codes(inputs)
        checksum ^= int(outputs[0])
        checksum ^= int(outputs[-1])
    elapsed_ns = time.perf_counter_ns() - start_ns
    return elapsed_ns / ITERATIONS, checksum


def time_mixed_forward_with_overflow(
    compiled: MixedDenseRunner,
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


def main() -> int:
    load_average_before = load_average()
    weights, inputs = deterministic_inputs()
    mixed_format = QFormatMixed(scale_per_tensor=False)
    compiled = compile_dense_mixed_precision(weights, fmt=mixed_format)

    mixed_results = [time_mixed_forward(compiled, inputs) for _ in range(REPEATS)]
    overflow_results = [time_mixed_forward_with_overflow(compiled, inputs) for _ in range(REPEATS)]
    float_results = [time_float_dot(weights, inputs) for _ in range(REPEATS)]
    mixed_ns = [float(item[0]) for item in mixed_results]
    overflow_ns = [float(item[0]) for item in overflow_results]
    float_ns = [float(item[0]) for item in float_results]
    reconstructed = compiled.forward_float(inputs)
    reference = weights @ inputs
    _, safe_overflow = compiled.forward_with_overflow(inputs)
    safe_envelope = compiled.precision_envelope_report(inputs)
    overflow_probe = compile_dense_mixed_precision(
        np.full((N_OUTPUTS, N_INPUTS), 127.0, dtype=np.float64),
        fmt=mixed_format,
    )
    _, probe_overflow = overflow_probe.forward_with_overflow(
        np.full(N_INPUTS, 32767.0, dtype=np.float64)
    )
    probe_envelope = overflow_probe.precision_envelope_report(
        np.full(N_INPUTS, 32767.0, dtype=np.float64)
    )

    report = {
        "benchmark": "mixed_dense_q88_q1616_64x32",
        "benchmark_contract": "canonical_q88_weight_q1616_input",
        "scale_per_tensor": False,
        "language": "Python",
        "timestamp_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "command": (
            "taskset -c 8-9 env PYTHONPATH=src "
            ".venv/bin/python benchmarks/bench_mixed_precision_dense.py"
        ),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "measurement_context": measurement_context(load_average_before),
        "n_inputs": N_INPUTS,
        "n_outputs": N_OUTPUTS,
        "iterations": ITERATIONS,
        "repeats": REPEATS,
        "mixed_median_ns_per_call": statistics.median(mixed_ns),
        "mixed_min_ns_per_call": min(mixed_ns),
        "mixed_max_ns_per_call": max(mixed_ns),
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
        "safe_required_total_bits": safe_envelope.required_total_bits,
        "safe_required_integer_bits": safe_envelope.required_integer_bits,
        "safe_width_headroom_bits": safe_envelope.width_headroom_bits,
        "safe_saturation_required": safe_envelope.saturation_required,
        "safe_static_overflow_proven_safe": safe_envelope.static_overflow_proven_safe,
        "saturating_probe_overflow_count": int(np.count_nonzero(probe_overflow)),
        "saturating_probe_max_abs_bound_code": probe_envelope.max_abs_bound_code,
        "saturating_probe_conservative_overflow_free": probe_envelope.conservative_overflow_free,
        "saturating_probe_required_total_bits": probe_envelope.required_total_bits,
        "saturating_probe_required_integer_bits": probe_envelope.required_integer_bits,
        "saturating_probe_width_headroom_bits": probe_envelope.width_headroom_bits,
        "saturating_probe_saturation_required": probe_envelope.saturation_required,
        "saturating_probe_static_overflow_proven_safe": (
            probe_envelope.static_overflow_proven_safe
        ),
        "compiled_manifest": compiled.manifest(),
        "mixed_results": [
            {"ns_per_call": result[0], "checksum": result[1]} for result in mixed_results
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
