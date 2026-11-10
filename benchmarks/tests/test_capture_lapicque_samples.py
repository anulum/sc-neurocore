# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Complete Lapicque raw-duration command and real report failures

"""Exercise the unchanged five-language workload through the public replay CLI."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
from typing import cast

import pytest

_BACKENDS = ("python", "rust", "julia", "go", "mojo")
_HARNESS_SHA256 = "0c709afd6567d7102ef828717b9399f91204300084b58042f94644e4df1e736a"


def _object(value: object) -> dict[str, object]:
    """Narrow one real JSON object after checking its key type."""
    assert isinstance(value, dict) and all(isinstance(key, str) for key in value)
    return cast(dict[str, object], value)


def _read(path: Path) -> dict[str, object]:
    """Read a result produced by the real command."""
    value: object = json.loads(path.read_text())
    return _object(value)


def _run(
    root: Path, output: Path, mode: str
) -> tuple[subprocess.CompletedProcess[str], dict[str, object]]:
    """Run full source-defined work or a genuine admission refusal in a child."""
    source = Path(__file__).resolve().parents[1] / "capture_lapicque_samples.py"
    harness = root / "benchmarks/bench_model_lapicque.py"
    assert hashlib.sha256(harness.read_bytes()).hexdigest() == _HARNESS_SHA256
    status = output.parent / (output.name + "-driver-status.json")
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1")
    result = subprocess.run(  # nosec B603 - actual repository CLI and audited fixed driver
        [
            sys.executable,
            "-B",
            str(Path(__file__).with_name("lapicque_cli_driver.py")),
            str(source),
            str(root),
            str(output),
            mode,
            str(status),
        ],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert hashlib.sha256(harness.read_bytes()).hexdigest() == _HARNESS_SHA256
    return result, _read(status)


def _assert_samples(record: dict[str, object]) -> dict[str, dict[str, object]]:
    """Validate all five real seven-sample lists and their reported aggregates."""
    raw = record["rows"]
    assert isinstance(raw, list)
    rows = [_object(row) for row in raw]
    assert [row["backend"] for row in rows] == list(_BACKENDS)
    by_backend: dict[str, dict[str, object]] = {}
    for row in rows:
        values = row["samples_ms"]
        assert isinstance(values, list) and len(values) == 7
        assert all(
            isinstance(value, float) and math.isfinite(value) and value > 0 for value in values
        )
        samples = [value for value in values if isinstance(value, float)]
        assert row["sample_count"] == 7
        assert row["median_call_ms"] == statistics.median(samples)
        assert row["minimum_call_ms"] == min(samples)
        assert row["original_elapsed_list_copied_after_timing"] is True
        by_backend[str(row["backend"])] = row
    assert record["original_harness_sha256"] == _HARNESS_SHA256
    assert record["source_and_function_identity_unchanged"] is True
    assert record["timed_functions_replaced"] is False
    assert record["production_performance_claim"] is False
    return by_backend


def _assert_report(report: dict[str, object], rows: dict[str, dict[str, object]]) -> None:
    """Check original workload, lane completeness, parity and raw aggregate identity."""
    workload = _object(report["workload"])
    assert workload["n_steps"] == 100_000 and workload["repeats"] == 7
    backends = _object(report["backends"])
    assert set(backends) == set(_BACKENDS)
    for name in _BACKENDS:
        lane = _object(backends[name])
        assert lane["available"] is True and lane["used"] is True
        assert lane["median_call_ms"] == rows[name]["median_call_ms"]
        assert lane["minimum_call_ms"] == rows[name]["minimum_call_ms"]
        assert lane["event_vector_matches_python"] is True
        error = lane["parity_max_abs_diff"]
        tolerance = workload["trace_atol"]
        assert isinstance(error, (float, int)) and isinstance(tolerance, (float, int))
        assert error <= tolerance


@pytest.mark.parametrize("mode", ["normal", "diagnostic", "outer-exception"])
def test_complete_original_cli_retains_every_sample(
    pytestconfig: pytest.Config, tmp_path: Path, mode: str
) -> None:
    """All five real lanes retain their original samples, including outer-except calls."""
    output = tmp_path / mode
    result, status = _run(Path(pytestconfig.rootpath), output, mode)
    assert result.returncode == 0, result.stdout + result.stderr
    record = _read(output / "original-seven-duration-samples.json")
    rows = _assert_samples(record)
    _assert_report(_read(output / "original-harness-report.json"), rows)
    assert record["raw_retention_valid"] is True
    assert record["original_report_aggregates_match_raw_samples"] is True
    assert record["execution_exception_type"] is None
    assert record["original_main_exit_code"] == 0
    assert record["observer_errors"] == []
    assert record["diagnostic_mode"] is (mode == "diagnostic")
    assert record["performance_admissible"] is (mode != "diagnostic")
    assert status["slot_after_cli"] is None
    assert status["profile_callback_preserved"] is True


@pytest.mark.parametrize(
    "mode, exception", [("corrupt-report", "JSONDecodeError"), ("partial-report", "KeyError")]
)
def test_actual_report_damage_retains_samples_and_fails_validation(
    pytestconfig: pytest.Config, tmp_path: Path, mode: str, exception: str
) -> None:
    """Real post-run file damage preserves the raw data and rejects the result."""
    output = tmp_path / mode
    result, status = _run(Path(pytestconfig.rootpath), output, mode)
    assert result.returncode == 1, result.stdout + result.stderr
    record = _read(output / "original-seven-duration-samples.json")
    rows = _assert_samples(record)
    _assert_report(_read(output / "unmodified-harness-report.json"), rows)
    assert record["raw_retention_valid"] is False
    assert record["performance_admissible"] is False
    assert record["original_main_exit_code"] == 0
    assert record["execution_exception_type"] is None
    errors = record["observer_errors"]
    assert isinstance(errors, list) and any(exception in str(error) for error in errors)
    assert status["slot_after_cli"] is None
    assert status["profile_callback_preserved"] is True


def test_original_harness_io_exception_propagates_after_retention(
    pytestconfig: pytest.Config, tmp_path: Path
) -> None:
    """An actual directory collision fails the unchanged harness after all five lanes."""
    output = tmp_path / "harness-failure"
    result, status = _run(Path(pytestconfig.rootpath), output, "harness-io-failure")
    assert result.returncode != 0 and "IsADirectoryError" in result.stderr
    record = _read(output / "original-seven-duration-samples.json")
    _assert_samples(record)
    assert record["execution_exception_type"] == "IsADirectoryError"
    assert record["original_main_exit_code"] is None
    assert record["raw_retention_valid"] is False
    assert record["performance_admissible"] is False
    assert (output / "original-harness-report.json").is_dir()
    assert status["slot_after_cli"] is None


def test_retention_collision_preserves_original_exception_and_existing_bytes(
    pytestconfig: pytest.Config, tmp_path: Path
) -> None:
    """A second real write failure keeps the harness exception primary with a note."""
    output = tmp_path / "retention-failure"
    result, status = _run(Path(pytestconfig.rootpath), output, "retention-collision")
    assert result.returncode != 0 and "IsADirectoryError" in result.stderr
    assert "raw-retention record could not be written" in result.stderr
    assert (output / "original-seven-duration-samples.json").read_text() == "preserved sentinel"
    events = status["actual_harness_events"]
    assert isinstance(events, list)
    assert [_object(row)["backend"] for row in events] == list(_BACKENDS)
    assert status["slot_after_cli"] is None


@pytest.mark.parametrize("mode", ["occupied-slot", "refuse-profile"])
def test_existing_instrumentation_is_preserved_and_measurement_refused(
    pytestconfig: pytest.Config, tmp_path: Path, mode: str
) -> None:
    """Actual foreign monitoring or profiling prevents measurement without displacement."""
    output = tmp_path / mode
    result, status = _run(Path(pytestconfig.rootpath), output, mode)
    assert result.returncode != 0 and "RuntimeError" in result.stderr
    record = _read(output / "original-seven-duration-samples.json")
    assert record["raw_retention_valid"] is False
    assert record["performance_admissible"] is False
    assert record["rows"] == []
    assert record["execution_exception_type"] == "RuntimeError"
    assert not (output / "original-harness-report.json").exists()
    assert status["actual_harness_events"] == []
    assert status["foreign_slot_preserved"] is True
    assert status["profile_callback_preserved"] is True
