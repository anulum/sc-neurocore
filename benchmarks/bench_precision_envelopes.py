# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Precision envelope benchmark artefact writer

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
    QFormatMixed,
    PrecisionEnvelopeReport,
    compile_dense_block_floating,
    compile_dense_mixed_precision,
)


N_INPUTS = 64
N_OUTPUTS = 32
ITERATIONS = 2_000
REPEATS = 7
OUTPUT = Path("benchmarks/results/local_python_2026-06-04_precision_envelopes.json")


class EnvelopeReporter(Protocol):
    """Compiled dense object that can emit a precision envelope report."""

    def precision_envelope_report(self, inputs: np.ndarray) -> PrecisionEnvelopeReport:
        """Return conservative output range telemetry."""
        ...


def deterministic_safe_workloads() -> tuple[
    EnvelopeReporter, np.ndarray, EnvelopeReporter, np.ndarray
]:
    mixed_weights = np.array(
        [((idx * 17 + 11) % 513 - 256) / 256.0 for idx in range(N_INPUTS * N_OUTPUTS)],
        dtype=np.float64,
    ).reshape(N_OUTPUTS, N_INPUTS)
    bfp_weights = np.array(
        [((idx * 23 + 3) % 1025 - 512) for idx in range(N_INPUTS * N_OUTPUTS)],
        dtype=np.float64,
    ).reshape(N_OUTPUTS, N_INPUTS)
    inputs = np.array(
        [(((idx * 19 + 5) % 257 - 128) << 6) / 65536.0 for idx in range(N_INPUTS)],
        dtype=np.float64,
    )
    mixed = compile_dense_mixed_precision(
        mixed_weights,
        fmt=QFormatMixed(scale_per_tensor=False),
    )
    block_floating = compile_dense_block_floating(bfp_weights, fmt="BFP16E3X32")
    return mixed, inputs, block_floating, inputs


def deterministic_underflow_workloads() -> tuple[
    EnvelopeReporter, np.ndarray, EnvelopeReporter, np.ndarray
]:
    mixed_weights = np.zeros((N_OUTPUTS, N_INPUTS), dtype=np.float64)
    bfp_weights = np.zeros((N_OUTPUTS, N_INPUTS), dtype=np.float64)
    mixed_weights[:, 0] = 1.0 / 256.0
    bfp_weights[:, 0] = 0.125
    inputs = np.full(N_INPUTS, 1.0 / 65536.0, dtype=np.float64)
    mixed = compile_dense_mixed_precision(
        mixed_weights,
        fmt=QFormatMixed(scale_per_tensor=False),
    )
    block_floating = compile_dense_block_floating(bfp_weights, fmt="BFP16E3X32")
    return mixed, inputs, block_floating, inputs


def time_envelope_report(
    compiled: EnvelopeReporter,
    inputs: np.ndarray,
) -> tuple[float, int, bool, bool]:
    start_ns = time.perf_counter_ns()
    checksum = 0
    conservative_safe = False
    underflow_free = False
    for _ in range(ITERATIONS):
        report = compiled.precision_envelope_report(inputs)
        checksum ^= int(report.max_abs_bound_code)
        checksum ^= int(report.min_headroom_code)
        checksum ^= int(report.underflow_count)
        conservative_safe = bool(report.conservative_overflow_free)
        underflow_free = bool(report.observed_underflow_free)
    elapsed_ns = time.perf_counter_ns() - start_ns
    return elapsed_ns / ITERATIONS, checksum, conservative_safe, underflow_free


def main() -> int:
    load_average_before = load_average()
    mixed, mixed_inputs, block_floating, bfp_inputs = deterministic_safe_workloads()
    underflow_mixed, underflow_mixed_inputs, underflow_bfp, underflow_bfp_inputs = (
        deterministic_underflow_workloads()
    )
    mixed_results = [time_envelope_report(mixed, mixed_inputs) for _ in range(REPEATS)]
    bfp_results = [time_envelope_report(block_floating, bfp_inputs) for _ in range(REPEATS)]
    mixed_ns = [float(item[0]) for item in mixed_results]
    bfp_ns = [float(item[0]) for item in bfp_results]
    mixed_report = mixed.precision_envelope_report(mixed_inputs)
    bfp_report = block_floating.precision_envelope_report(bfp_inputs)
    mixed_underflow_report = underflow_mixed.precision_envelope_report(underflow_mixed_inputs)
    bfp_underflow_report = underflow_bfp.precision_envelope_report(underflow_bfp_inputs)

    report = {
        "benchmark": "precision_envelope_reports_64x32",
        "language": "Python",
        "timestamp_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "command": (
            "taskset -c 8-9 env PYTHONPATH=src "
            ".venv/bin/python benchmarks/bench_precision_envelopes.py"
        ),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "measurement_context": measurement_context(load_average_before),
        "n_inputs": N_INPUTS,
        "n_outputs": N_OUTPUTS,
        "iterations": ITERATIONS,
        "repeats": REPEATS,
        "mixed_envelope_median_ns_per_call": statistics.median(mixed_ns),
        "mixed_envelope_min_ns_per_call": min(mixed_ns),
        "mixed_envelope_max_ns_per_call": max(mixed_ns),
        "mixed_conservative_overflow_free": mixed_report.conservative_overflow_free,
        "mixed_observed_underflow_free": mixed_report.observed_underflow_free,
        "mixed_max_abs_bound_code": mixed_report.max_abs_bound_code,
        "mixed_required_total_bits": mixed_report.required_total_bits,
        "mixed_required_integer_bits": mixed_report.required_integer_bits,
        "mixed_width_headroom_bits": mixed_report.width_headroom_bits,
        "mixed_saturation_required": mixed_report.saturation_required,
        "mixed_static_overflow_proven_safe": mixed_report.static_overflow_proven_safe,
        "mixed_underflow_count": mixed_underflow_report.underflow_count,
        "bfp_envelope_median_ns_per_call": statistics.median(bfp_ns),
        "bfp_envelope_min_ns_per_call": min(bfp_ns),
        "bfp_envelope_max_ns_per_call": max(bfp_ns),
        "bfp_conservative_overflow_free": bfp_report.conservative_overflow_free,
        "bfp_observed_underflow_free": bfp_report.observed_underflow_free,
        "bfp_max_abs_bound_code": bfp_report.max_abs_bound_code,
        "bfp_required_total_bits": bfp_report.required_total_bits,
        "bfp_required_integer_bits": bfp_report.required_integer_bits,
        "bfp_width_headroom_bits": bfp_report.width_headroom_bits,
        "bfp_saturation_required": bfp_report.saturation_required,
        "bfp_static_overflow_proven_safe": bfp_report.static_overflow_proven_safe,
        "bfp_underflow_count": bfp_underflow_report.underflow_count,
        "mixed_manifest": mixed_report.manifest(),
        "bfp_manifest": bfp_report.manifest(),
        "mixed_underflow_manifest": mixed_underflow_report.manifest(),
        "bfp_underflow_manifest": bfp_underflow_report.manifest(),
        "mixed_results": [
            {
                "ns_per_call": item[0],
                "checksum": item[1],
                "conservative_safe": item[2],
                "underflow_free": item[3],
            }
            for item in mixed_results
        ],
        "bfp_results": [
            {
                "ns_per_call": item[0],
                "checksum": item[1],
                "conservative_safe": item[2],
                "underflow_free": item[3],
            }
            for item in bfp_results
        ],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
