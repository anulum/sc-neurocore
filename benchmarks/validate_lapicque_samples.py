# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Validate original Lapicque samples before cohort comparison

"""Reject diagnostic or altered timing evidence before comparing five language lanes."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from typing import cast

_BACKENDS = ("python", "rust", "julia", "go", "mojo")
_HARNESS = "benchmarks/bench_model_lapicque.py"
_HARNESS_SHA256 = "0c709afd6567d7102ef828717b9399f91204300084b58042f94644e4df1e736a"


def _object(value: object) -> dict[str, object]:
    """Require an actual JSON object with string keys."""
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError("expected a JSON object")
    return cast(dict[str, object], value)


def _require(condition: bool, reason: str) -> None:
    """Reject a violated data or provenance invariant."""
    if not condition:
        raise ValueError(reason)


def _number(value: object) -> float:
    """Read a finite JSON number without accepting Boolean flags."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("expected a finite numeric measurement")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError("numeric measurement exceeds the finite range") from error
    _require(math.isfinite(result), "non-finite measurement")
    return result


def validate_samples(checkout: Path, samples_path: Path, report_path: Path) -> dict[str, object]:
    """Validate raw durations and their original report before cohort comparison.

    Parameters
    ----------
    checkout : pathlib.Path
        Checkout containing the exact frozen benchmark and its declared sources.
    samples_path : pathlib.Path
        Original duration-retention JSON emitted by the adjacent capture command.
    report_path : pathlib.Path
        Unchanged JSON report written by the original five-language harness.

    Returns
    -------
    dict[str, object]
        Verified sample aggregates and identity hashes. External runtime provenance
        and physical isolation remain required for performance acceptance.

    Raises
    ------
    OSError
        An input cannot be read.
    ValueError
        Diagnostic mode, invalid data, source drift or aggregate drift is detected.
    """
    root = checkout.resolve(strict=True)
    raw_samples = samples_path.read_bytes()
    samples = _object(json.loads(raw_samples))
    _require(
        samples.get("performance_admissible") is True, "timings are not performance-admissible"
    )
    _require(
        samples.get("isolation_acceptance_requires_external_supervision") is True,
        "external performance qualification boundary is absent",
    )
    _require(
        samples.get("schema_version") == "lapicque-original-duration-observations.v1",
        "unknown sample schema",
    )
    for key in (
        "raw_retention_valid",
        "no_instrumentation_observed_at_boundaries",
        "source_and_function_identity_unchanged",
        "original_report_aggregates_match_raw_samples",
    ):
        _require(samples.get(key) is True, "invalid sample evidence: " + key)
    for key in (
        "diagnostic_mode",
        "timed_functions_replaced",
        "global_monitoring_events_enabled",
        "production_performance_claim",
    ):
        _require(samples.get(key) is False, "inadmissible sample evidence: " + key)
    for key in ("instrumentation_at_install", "instrumentation_at_close", "observer_errors"):
        _require(samples.get(key) == [], "instrumentation or observer errors remain: " + key)
    _require(
        samples.get("execution_exception_type") is None, "original execution raised an exception"
    )
    exit_code = samples.get("original_main_exit_code")
    _require(type(exit_code) is int and exit_code == 0, "original harness did not succeed")
    harness_raw = (root / _HARNESS).read_bytes()
    _require(
        hashlib.sha256(harness_raw).hexdigest() == _HARNESS_SHA256, "frozen harness source changed"
    )
    _require(samples.get("original_harness") == _HARNESS, "unexpected original harness")
    _require(
        samples.get("original_harness_sha256") == _HARNESS_SHA256,
        "sample harness identity mismatch",
    )
    observer = Path(__file__).with_name("capture_lapicque_samples.py")
    _require(
        samples.get("observer_sha256") == hashlib.sha256(observer.read_bytes()).hexdigest(),
        "capture source identity mismatch",
    )
    raw_report = report_path.read_bytes()
    _require(
        samples.get("original_report_sha256") == hashlib.sha256(raw_report).hexdigest(),
        "original report bytes changed",
    )
    report = _object(json.loads(raw_report))
    _require(
        report.get("schema_version") == "sc-neurocore.polyglot-benchmark.v1",
        "unknown harness report schema",
    )
    _require(
        report.get("kernel") == "lapicque_1907_exact_polarization_complete_packet",
        "unexpected model kernel",
    )
    meta = _object(report.get("meta"))
    affinity = meta.get("affinity")
    _require(
        meta.get("single_cpu_pinned") is True
        and isinstance(affinity, list)
        and len(affinity) == 1
        and type(affinity[0]) is int
        and affinity[0] >= 0,
        "original single-CPU workload admission is absent",
    )
    workload = _object(report.get("workload"))
    _require(
        type(workload.get("n_steps")) is int and workload["n_steps"] == 100_000,
        "original step count changed",
    )
    _require(
        type(workload.get("repeats")) is int and workload["repeats"] == 7,
        "original repetition count changed",
    )
    _require(_number(workload.get("trace_atol")) == 5e-15, "original parity bound changed")
    _require(_number(workload.get("source_voltage")) == 22.0, "original source voltage changed")
    source_fields = _object(report.get("source_hashes"))
    assignments = [node for node in ast.parse(harness_raw).body if isinstance(node, ast.Assign)]
    declarations = [
        ast.literal_eval(node.value)
        for node in assignments
        if any(
            isinstance(target, ast.Name) and target.id == "SOURCE_PATHS" for target in node.targets
        )
    ]
    _require(len(declarations) == 1, "frozen source inventory is unavailable")
    for relative in declarations[0]:
        actual = (root / relative).read_bytes()
        _require(
            source_fields.get(relative) == hashlib.sha256(actual).hexdigest(),
            "declared benchmark input changed: " + relative,
        )
    raw_rows = samples.get("rows")
    _require(isinstance(raw_rows, list), "sample rows are absent")
    rows = [_object(row) for row in cast(list[object], raw_rows)]
    _require(
        tuple(row.get("backend") for row in rows) == _BACKENDS,
        "incomplete or repeated language lanes",
    )
    lanes = _object(report.get("backends"))
    _require(set(lanes) == set(_BACKENDS), "original report language lanes changed")
    summaries: list[dict[str, object]] = []
    python_median = 0.0
    for row in rows:
        backend = str(row["backend"])
        _require(
            row.get("instrumentation_at_return") == [],
            "instrumentation observed at measurement return",
        )
        _require(
            row.get("original_elapsed_list_copied_after_timing") is True,
            "raw samples lack original-return observation",
        )
        values = row.get("samples_ms")
        _require(isinstance(values, list), "duration list is absent")
        durations = [_number(value) for value in cast(list[object], values)]
        _require(
            len(durations) == 7 and all(value > 0 for value in durations),
            "seven positive durations are required",
        )
        _require(
            type(row.get("sample_count")) is int and row["sample_count"] == 7,
            "sample count changed",
        )
        median = statistics.median(durations)
        minimum = min(durations)
        _require(
            _number(row.get("median_call_ms")) == median,
            "retained median disagrees with raw samples",
        )
        _require(
            _number(row.get("minimum_call_ms")) == minimum,
            "retained minimum disagrees with raw samples",
        )
        lane = _object(lanes[backend])
        _require(
            lane.get("available") is True and lane.get("used") is True, "unexecuted language lane"
        )
        _require(
            _number(lane.get("median_call_ms")) == median,
            "report median disagrees with raw samples",
        )
        _require(
            _number(lane.get("minimum_call_ms")) == minimum,
            "report minimum disagrees with raw samples",
        )
        _require(lane.get("event_vector_matches_python") is True, "event-vector parity failed")
        _require(
            0 <= _number(lane.get("parity_max_abs_diff")) <= 5e-15, "voltage-trace parity failed"
        )
        if backend == "python":
            python_median = median
        speedup = python_median / median
        _require(
            _number(lane.get("speedup_vs_python")) == speedup,
            "report speedup disagrees with raw samples",
        )
        summaries.append(
            dict(
                backend=backend,
                sample_count=7,
                median_call_ms=median,
                minimum_call_ms=minimum,
                speedup_vs_python=speedup,
            )
        )
    return dict(
        schema_version="lapicque-validated-duration-samples.v1",
        samples_sha256=hashlib.sha256(raw_samples).hexdigest(),
        report_sha256=hashlib.sha256(raw_report).hexdigest(),
        harness_sha256=_HARNESS_SHA256,
        sample_integrity_verified=True,
        external_isolation_and_runtime_provenance_required=True,
        production_performance_claim=False,
        lanes=summaries,
    )


def main(argv: list[str] | None = None) -> int:
    """Validate complete original timing evidence through the public CLI.

    Parameters
    ----------
    argv : list of str or None
        Explicit input paths, or process arguments when omitted.

    Returns
    -------
    int
        Zero for consistent admissible samples; two for rejected evidence.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", required=True, type=Path)
    parser.add_argument("--samples", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = validate_samples(args.checkout, args.samples, args.report)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(str(error), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
