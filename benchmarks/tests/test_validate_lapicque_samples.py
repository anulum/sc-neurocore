# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Public validation of actual timing evidence and altered copies

"""Validate genuine complete-run samples and reject changes to their evidence."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from collections.abc import Callable
from pathlib import Path
import statistics
import subprocess
import sys
from typing import cast

import pytest


def _object(value: object) -> dict[str, object]:
    """Narrow an actual decoded JSON object."""
    assert isinstance(value, dict) and all(isinstance(key, str) for key in value)
    return cast(dict[str, object], value)


def _read(path: Path) -> dict[str, object]:
    """Read one real producer output or an explicitly altered negative input."""
    value: object = json.loads(path.read_text())
    return _object(value)


def _command(samples: Path, report: Path, checkout: Path) -> subprocess.CompletedProcess[str]:
    """Invoke the actual validation CLI without bypassing any check."""
    script = Path(__file__).resolve().parents[1] / "validate_lapicque_samples.py"
    return subprocess.run(  # nosec B603 - fixed repository CLI and explicit evidence paths
        [
            sys.executable,
            "-B",
            str(script),
            "--samples",
            str(samples),
            "--report",
            str(report),
            "--checkout",
            str(checkout),
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def _changed_input(tmp_path: Path, original: Path, payload: dict[str, object]) -> Path:
    """Write an altered copy while leaving the actual producer artifact intact."""
    output = tmp_path / original.name
    with output.open("x") as stream:
        json.dump(payload, stream)
    return output


def test_actual_original_durations_match_the_validated_language_rows(
    original_lapicque_evidence: tuple[Path, Path, Path],
) -> None:
    """The public consumer accepts genuine data and reproduces its aggregates."""
    samples, report, checkout = original_lapicque_evidence
    before = [hashlib.sha256(path.read_bytes()).hexdigest() for path in (samples, report)]
    result = _command(samples, report, checkout)
    assert result.returncode == 0, result.stderr
    validated = _object(json.loads(result.stdout))
    assert validated["sample_integrity_verified"] is True
    assert validated["production_performance_claim"] is False
    assert validated["external_isolation_and_runtime_provenance_required"] is True
    raw_rows = _read(samples)["rows"]
    output_rows = validated["lanes"]
    assert isinstance(raw_rows, list) and isinstance(output_rows, list)
    assert len(raw_rows) == len(output_rows) == 5
    for raw, output in zip(raw_rows, output_rows, strict=True):
        row, actual = _object(raw), _object(output)
        values = row["samples_ms"]
        assert isinstance(values, list)
        numbers = [float(value) for value in values]
        assert actual["backend"] == row["backend"]
        assert actual["sample_count"] == len(numbers) == 7
        assert actual["median_call_ms"] == statistics.median(numbers)
        assert actual["minimum_call_ms"] == min(numbers)
    assert [hashlib.sha256(path.read_bytes()).hexdigest() for path in (samples, report)] == before


def test_public_python_api_validates_the_same_actual_evidence(
    original_lapicque_evidence: tuple[Path, Path, Path],
) -> None:
    """The documented public function reads the same real files as the CLI."""
    source = Path(__file__).resolve().parents[1] / "validate_lapicque_samples.py"
    spec = importlib.util.spec_from_file_location("lapicque_duration_validation", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    validate = cast(Callable[[Path, Path, Path], dict[str, object]], module.validate_samples)
    samples, report, checkout = original_lapicque_evidence
    result = validate(checkout, samples, report)
    assert result["sample_integrity_verified"] is True
    assert result["samples_sha256"] == hashlib.sha256(samples.read_bytes()).hexdigest()


def test_language_rows_cannot_replace_the_retention_envelope(
    original_lapicque_evidence: tuple[Path, Path, Path],
    tmp_path: Path,
) -> None:
    """A bare array of genuine rows lacks the required eligibility envelope."""
    samples, report, checkout = original_lapicque_evidence
    changed = tmp_path / samples.name
    changed.write_text(json.dumps(_read(samples)["rows"]))
    result = _command(changed, report, checkout)
    assert result.returncode == 2 and "expected a JSON object" in result.stderr


@pytest.mark.parametrize(
    "field,value",
    [
        ("performance_admissible", False),
        ("diagnostic_mode", True),
        ("raw_retention_valid", False),
        ("no_instrumentation_observed_at_boundaries", False),
        ("source_and_function_identity_unchanged", False),
        ("original_report_aggregates_match_raw_samples", False),
        ("timed_functions_replaced", True),
        ("global_monitoring_events_enabled", True),
        ("production_performance_claim", True),
        ("isolation_acceptance_requires_external_supervision", False),
        ("instrumentation_at_install", ["profiling"]),
        ("instrumentation_at_close", ["profiling"]),
        ("observer_errors", ["capture failed"]),
        ("execution_exception_type", "OSError"),
        ("original_main_exit_code", True),
        ("schema_version", "unknown"),
        ("original_harness", "other.py"),
        ("original_harness_sha256", "0" * 64),
        ("observer_sha256", "0" * 64),
    ],
)
def test_inadmissible_or_altered_capture_metadata_is_rejected(
    original_lapicque_evidence: tuple[Path, Path, Path],
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    """Unsafe eligibility or provenance fields cannot enter a comparison."""
    samples, report, checkout = original_lapicque_evidence
    payload = _read(samples)
    payload[field] = value
    changed = _changed_input(tmp_path, samples, payload)
    result = _command(changed, report, checkout)
    assert result.returncode == 2 and not result.stdout


@pytest.mark.parametrize("replacement", [True, "not a duration", float("nan"), 0.0, -1.0, 10**400])
def test_nonfinite_nonnumeric_or_nonpositive_durations_are_rejected(
    original_lapicque_evidence: tuple[Path, Path, Path],
    tmp_path: Path,
    replacement: object,
) -> None:
    """Actual sample copies with invalid duration values fail the public CLI."""
    samples, report, checkout = original_lapicque_evidence
    payload = _read(samples)
    rows = payload["rows"]
    assert isinstance(rows, list)
    values = _object(rows[0])["samples_ms"]
    assert isinstance(values, list)
    values[0] = replacement
    result = _command(_changed_input(tmp_path, samples, payload), report, checkout)
    assert result.returncode == 2 and not result.stdout


@pytest.mark.parametrize(
    "mutation",
    [
        "missing-row",
        "repeated-row",
        "count",
        "median",
        "minimum",
        "instrumentation",
        "return-proof",
    ],
)
def test_missing_or_inconsistent_language_observations_are_rejected(
    original_lapicque_evidence: tuple[Path, Path, Path],
    tmp_path: Path,
    mutation: str,
) -> None:
    """Every language row must preserve its real list, return boundary and aggregates."""
    samples, report, checkout = original_lapicque_evidence
    payload = _read(samples)
    rows = payload["rows"]
    assert isinstance(rows, list)
    row = _object(rows[0])
    if mutation == "missing-row":
        rows.pop()
    elif mutation == "repeated-row":
        rows[-1] = rows[0]
    else:
        field, value = {
            "count": ("sample_count", True),
            "median": ("median_call_ms", -1.0),
            "minimum": ("minimum_call_ms", -1.0),
            "instrumentation": ("instrumentation_at_return", ["profiling"]),
            "return-proof": ("original_elapsed_list_copied_after_timing", False),
        }[mutation]
        row[field] = value
    result = _command(_changed_input(tmp_path, samples, payload), report, checkout)
    assert result.returncode == 2 and not result.stdout


def test_changed_original_report_bytes_are_rejected(
    original_lapicque_evidence: tuple[Path, Path, Path],
    tmp_path: Path,
) -> None:
    """Even unchanged report values cannot hide changed original bytes."""
    samples, report, checkout = original_lapicque_evidence
    changed = tmp_path / report.name
    changed.write_bytes(report.read_bytes() + b"\n")
    result = _command(samples, changed, checkout)
    assert result.returncode == 2 and "report bytes changed" in result.stderr


@pytest.mark.parametrize(
    "field,value",
    [
        ("used", False),
        ("available", False),
        ("median_call_ms", -1.0),
        ("minimum_call_ms", -1.0),
        ("speedup_vs_python", -1.0),
        ("event_vector_matches_python", False),
        ("parity_max_abs_diff", 1.0),
    ],
)
def test_changed_report_aggregates_or_parity_fail_after_hash_rebinding(
    original_lapicque_evidence: tuple[Path, Path, Path],
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    """A matching file hash cannot make altered metrics or failed parity acceptable."""
    samples, report, checkout = original_lapicque_evidence
    altered_report = _read(report)
    lanes = _object(altered_report["backends"])
    _object(lanes["python"])[field] = value
    report_copy = _changed_input(tmp_path, report, altered_report)
    payload = _read(samples)
    payload["original_report_sha256"] = hashlib.sha256(report_copy.read_bytes()).hexdigest()
    result = _command(_changed_input(tmp_path, samples, payload), report_copy, checkout)
    assert result.returncode == 2 and not result.stdout


def test_missing_actual_input_is_reported_without_a_success_payload(
    original_lapicque_evidence: tuple[Path, Path, Path],
    tmp_path: Path,
) -> None:
    """A nonexistent producer output fails through the same public command."""
    _, report, checkout = original_lapicque_evidence
    result = _command(tmp_path / "absent-samples.json", report, checkout)
    assert result.returncode == 2 and not result.stdout
