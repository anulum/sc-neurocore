# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Instrumented full-workload Lapicque command qualification

"""Exercise real diagnostic commands while tracing their capture implementation."""

from pathlib import Path

import pytest

from test_capture_lapicque_samples import _assert_report, _assert_samples, _read, _run


@pytest.mark.parametrize("mode", ["diagnostic", "outer-diagnostic", "api-diagnostic"])
def test_instrumented_full_workload_retains_all_original_samples(
    pytestconfig: pytest.Config, tmp_path: Path, mode: str
) -> None:
    """Diagnostic CLI and public API calls keep complete data without performance credit."""
    output = tmp_path / mode
    result, status = _run(Path(pytestconfig.rootpath), output, mode)
    assert result.returncode == 0, result.stdout + result.stderr
    record = _read(output / "original-seven-duration-samples.json")
    _assert_report(_read(output / "original-harness-report.json"), _assert_samples(record))
    assert record["raw_retention_valid"] is True
    assert record["diagnostic_mode"] is True
    assert record["performance_admissible"] is False
    assert record["execution_exception_type"] is None
    assert record["observer_errors"] == []
    assert status["slot_after_cli"] is None
    assert status["profile_callback_preserved"] is True


@pytest.mark.parametrize("mode", ["corrupt-report", "partial-report", "harness-io-failure"])
def test_genuine_output_failures_preserve_full_diagnostic_samples(
    pytestconfig: pytest.Config, tmp_path: Path, mode: str
) -> None:
    """Malformed reports or original write errors reject the complete measured run."""
    output = tmp_path / mode
    result, status = _run(Path(pytestconfig.rootpath), output, mode)
    assert result.returncode != 0
    record = _read(output / "original-seven-duration-samples.json")
    _assert_samples(record)
    assert record["raw_retention_valid"] is False
    assert record["performance_admissible"] is False
    if mode == "harness-io-failure":
        assert record["execution_exception_type"] == "IsADirectoryError"
        assert "IsADirectoryError" in result.stderr
    else:
        assert record["execution_exception_type"] is None
        assert record["original_main_exit_code"] == 0
        _assert_report(_read(output / "unmodified-harness-report.json"), _assert_samples(record))
    assert status["slot_after_cli"] is None


def test_second_write_failure_keeps_original_exception_and_existing_bytes(
    pytestconfig: pytest.Config, tmp_path: Path
) -> None:
    """Real report and retention collisions retain the original primary error."""
    output = tmp_path / "retention-collision"
    result, status = _run(Path(pytestconfig.rootpath), output, "retention-collision")
    assert result.returncode != 0
    assert "IsADirectoryError" in result.stderr
    assert "raw-retention record could not be written" in result.stderr
    assert (output / "original-seven-duration-samples.json").read_text() == "preserved sentinel"
    events = status["actual_harness_events"]
    assert isinstance(events, list) and len(events) == 5
    assert status["slot_after_cli"] is None


@pytest.mark.parametrize("mode", ["occupied-slot", "refuse-profile"])
def test_real_instrumentation_admission_preserves_foreign_owners(
    pytestconfig: pytest.Config, tmp_path: Path, mode: str
) -> None:
    """A real occupied slot or forbidden default profiler prevents measurement."""
    output = tmp_path / mode
    result, status = _run(Path(pytestconfig.rootpath), output, mode)
    assert result.returncode != 0 and "RuntimeError" in result.stderr
    record = _read(output / "original-seven-duration-samples.json")
    assert record["rows"] == []
    assert record["raw_retention_valid"] is False
    assert record["performance_admissible"] is False
    assert status["foreign_slot_preserved"] is True
    assert status["profile_callback_preserved"] is True
