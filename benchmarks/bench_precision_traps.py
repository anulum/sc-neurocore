# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Precision trap benchmark artefact writer

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
import platform
import statistics
import time
from typing import Protocol

import numpy as np

from sc_neurocore.compiler.quantizer import (
    QFormatMixed,
    PrecisionTrapReport,
    compile_dense_block_floating,
    compile_dense_mixed_precision,
)


N_INPUTS = 64
N_OUTPUTS = 32
ITERATIONS = 2_000
REPEATS = 7
OUTPUT = Path("benchmarks/results/local_python_2026-06-04_precision_traps.json")


class TrapReporter(Protocol):
    """Compiled dense object that can emit precision trap telemetry."""

    def precision_trap_report(self, inputs: np.ndarray) -> PrecisionTrapReport:
        """Return saturation telemetry for a fixed-point dense operation."""
        ...


def deterministic_overflow_workloads() -> tuple[TrapReporter, np.ndarray, TrapReporter, np.ndarray]:
    mixed_weights = np.full((N_OUTPUTS, N_INPUTS), 127.0, dtype=np.float64)
    bfp_weights = np.full((N_OUTPUTS, N_INPUTS), 8192.0, dtype=np.float64)
    inputs = np.full(N_INPUTS, 32767.0, dtype=np.float64)
    mixed = compile_dense_mixed_precision(
        mixed_weights,
        fmt=QFormatMixed(scale_per_tensor=False),
    )
    block_floating = compile_dense_block_floating(bfp_weights, fmt="BFP16E3X32")
    return mixed, inputs, block_floating, inputs


def time_trap_report(compiled: TrapReporter, inputs: np.ndarray) -> tuple[float, int, int]:
    start_ns = time.perf_counter_ns()
    checksum = 0
    overflow_count = 0
    for _ in range(ITERATIONS):
        report = compiled.precision_trap_report(inputs)
        checksum ^= int(report.saturated_max_count)
        checksum ^= int(report.saturated_min_count)
        overflow_count = int(report.overflow_count)
    elapsed_ns = time.perf_counter_ns() - start_ns
    return elapsed_ns / ITERATIONS, checksum, overflow_count


def main() -> int:
    mixed, mixed_inputs, block_floating, bfp_inputs = deterministic_overflow_workloads()
    mixed_results = [time_trap_report(mixed, mixed_inputs) for _ in range(REPEATS)]
    bfp_results = [time_trap_report(block_floating, bfp_inputs) for _ in range(REPEATS)]
    mixed_ns = [float(item[0]) for item in mixed_results]
    bfp_ns = [float(item[0]) for item in bfp_results]

    report = {
        "benchmark": "precision_trap_reports_64x32",
        "language": "Python",
        "timestamp_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "command": "PYTHONPATH=src .venv/bin/python benchmarks/bench_precision_traps.py",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "n_inputs": N_INPUTS,
        "n_outputs": N_OUTPUTS,
        "iterations": ITERATIONS,
        "repeats": REPEATS,
        "mixed_trap_median_ns_per_call": statistics.median(mixed_ns),
        "mixed_trap_min_ns_per_call": min(mixed_ns),
        "mixed_trap_max_ns_per_call": max(mixed_ns),
        "mixed_overflow_count": mixed_results[-1][2],
        "bfp_trap_median_ns_per_call": statistics.median(bfp_ns),
        "bfp_trap_min_ns_per_call": min(bfp_ns),
        "bfp_trap_max_ns_per_call": max(bfp_ns),
        "bfp_overflow_count": bfp_results[-1][2],
        "mixed_manifest": mixed.precision_trap_report(mixed_inputs).manifest(),
        "bfp_manifest": block_floating.precision_trap_report(bfp_inputs).manifest(),
        "mixed_results": [
            {"ns_per_call": item[0], "checksum": item[1], "overflow_count": item[2]}
            for item in mixed_results
        ],
        "bfp_results": [
            {"ns_per_call": item[0], "checksum": item[1], "overflow_count": item[2]}
            for item in bfp_results
        ],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
