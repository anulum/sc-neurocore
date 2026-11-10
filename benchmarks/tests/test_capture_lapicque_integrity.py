# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Real post-measurement source and output integrity faults


"""Verify that real post-computation integrity failures never qualify retained measurements."""

import math
import os
from pathlib import Path
import subprocess
import sys

import pytest

from test_capture_lapicque_samples import _assert_report, _object, _read


@pytest.mark.parametrize(
    "fault",
    ["source-removed", "backend-metadata", "report-read-failure", "retention-only-collision"],
)
def test_real_post_computation_integrity_failures_are_rejected(
    pytestconfig: pytest.Config, tmp_path: Path, fault: str
) -> None:
    """Full original dynamics complete before a real source, metadata or output failure."""
    original = Path(pytestconfig.rootpath)
    checkout = original
    if fault == "source-removed":
        checkout = tmp_path / "frozen-checkout"
        harness = checkout / "benchmarks/bench_model_lapicque.py"
        harness.parent.mkdir(parents=True)
        harness.write_bytes((original / "benchmarks/bench_model_lapicque.py").read_bytes())
        for name in ("src", "engine", "hdl", "bridge"):
            (checkout / name).symlink_to(original / name, target_is_directory=True)
    source = Path(__file__).resolve().parents[1] / "capture_lapicque_samples.py"
    output = tmp_path / fault
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            str(Path(__file__).with_name("lapicque_integrity_driver.py")),
            str(source),
            str(checkout),
            str(output),
            fault,
            str(tmp_path / "driver-status.json"),
        ],
        cwd=original,
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
        capture_output=True,
        text=True,
        check=False,
    )
    state = _read(tmp_path / "actual-integrity-state.json")
    assert state["changed_after_full_command"] is True
    assert state["actual_backend_returns"] == ["python", "rust", "julia", "go", "mojo"]
    assert state["profile_owner_preserved"] is True
    assert state["monitoring_owner_after_cli"] is None
    if fault == "retention-only-collision":
        assert result.returncode != 0 and "FileExistsError" in result.stderr
        assert "raw-retention record could not be written" not in result.stderr
        assert (output / "original-seven-duration-samples.json").read_text() == "preserved sentinel"
        report = _read(output / "original-harness-report.json")
        assert set(_object(report["backends"])) == {"python", "rust", "julia", "go", "mojo"}
        return
    assert result.returncode == 1, result.stdout + result.stderr
    record = _read(output / "original-seven-duration-samples.json")
    assert record["raw_retention_valid"] is False and record["performance_admissible"] is False
    assert record["original_main_exit_code"] == 0 and record["execution_exception_type"] is None
    raw_rows = record["rows"]
    assert isinstance(raw_rows, list) and len(raw_rows) == 5
    rows = [_object(row) for row in raw_rows]
    for row in rows:
        values = row["samples_ms"]
        assert isinstance(values, list) and len(values) == row["sample_count"] == 7
        assert all(
            isinstance(value, float) and math.isfinite(value) and value > 0 for value in values
        )
    errors = record["observer_errors"]
    assert isinstance(errors, list)
    if fault == "source-removed":
        assert record["source_and_function_identity_unchanged"] is False
        assert any("source validation failed: FileNotFoundError" in str(error) for error in errors)
        _assert_report(
            _read(output / "original-harness-report.json"),
            {str(row["backend"]): row for row in rows},
        )
    elif fault == "backend-metadata":
        assert record["source_and_function_identity_unchanged"] is True
        assert rows[0]["backend"] is False
        assert record["original_report_aggregates_match_raw_samples"] is False
    else:
        assert state["actual_report_read_fault"] is True
        assert any("report validation failed: IsADirectoryError" in str(error) for error in errors)
        _assert_report(
            _read(output / "original-harness-report.json"),
            {str(row["backend"]): row for row in rows},
        )
