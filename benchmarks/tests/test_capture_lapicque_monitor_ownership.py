# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Real monitor-state changes after full original benchmark work

"""Verify owned monitor cleanup and foreign ownership through the complete public CLI."""

import os
from pathlib import Path
import subprocess
import sys

import pytest

from test_capture_lapicque_samples import _assert_samples, _read


@pytest.mark.parametrize(
    "scenario", ["global-events", "local-events", "foreign-owner", "callback-owner"]
)
def test_monitor_state_changes_reject_capture_and_preserve_ownership(
    pytestconfig: pytest.Config, tmp_path: Path, scenario: str
) -> None:
    """Full genuine measurements are rejected after real monitor-state interference."""
    root = Path(pytestconfig.rootpath)
    source = Path(__file__).resolve().parents[1] / "capture_lapicque_samples.py"
    output = tmp_path / scenario
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            str(Path(__file__).with_name("lapicque_monitor_driver.py")),
            str(source),
            str(root),
            str(output),
            scenario,
            str(tmp_path / "driver-status.json"),
        ],
        cwd=root,
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    state = _read(tmp_path / "actual-monitor-state.json")
    record = _read(output / "original-seven-duration-samples.json")
    _assert_samples(record)
    assert state["changed_after_full_command"] is True
    assert state["profile_owner_preserved"] is True
    assert record["raw_retention_valid"] is False
    assert record["performance_admissible"] is False
    errors = record["observer_errors"]
    assert isinstance(errors, list)
    if scenario == "foreign-owner":
        assert state["owner_after_command"] == "lapicque-monitor-ownership-test"
        assert any("foreign slot retained" in str(error) for error in errors)
    else:
        assert state["owner_after_command"] is None
        assert any(
            (
                {
                    "global-events": "global monitoring",
                    "local-events": "local return-event",
                    "callback-owner": "return callback changed",
                }[scenario]
            )
            in str(error)
            for error in errors
        )
        assert state["global_events_after_command"] == 0


@pytest.mark.parametrize("scenario", ["stale-global", "stale-callback"])
def test_real_unclaimed_monitor_metadata_refuses_before_measurement(
    pytestconfig: pytest.Config, tmp_path: Path, scenario: str
) -> None:
    """A genuinely freed slot with stale configuration cannot be silently reused."""
    root = Path(pytestconfig.rootpath)
    source = Path(__file__).resolve().parents[1] / "capture_lapicque_samples.py"
    output = tmp_path / scenario
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            str(Path(__file__).with_name("lapicque_monitor_driver.py")),
            str(source),
            str(root),
            str(output),
            scenario,
            str(tmp_path / "driver-status.json"),
        ],
        cwd=root,
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0 and "monitoring slot was not empty" in result.stderr
    state = _read(tmp_path / "actual-monitor-state.json")
    record = _read(output / "original-seven-duration-samples.json")
    assert state["stale_before_command"] is True
    assert state["changed_after_full_command"] is False
    assert state["owner_after_command"] is None
    assert state["global_events_after_command"] == 0
    assert record["rows"] == []
    assert record["raw_retention_valid"] is False and record["performance_admissible"] is False


def test_actual_owner_loss_during_cleanup_rejects_complete_capture(
    pytestconfig: pytest.Config, tmp_path: Path
) -> None:
    """An actual audit hook frees the owned slot during callback removal after full work."""
    root = Path(pytestconfig.rootpath)
    source = Path(__file__).resolve().parents[1] / "capture_lapicque_samples.py"
    output = tmp_path / "cleanup-owner-loss"
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            str(Path(__file__).with_name("lapicque_monitor_driver.py")),
            str(source),
            str(root),
            str(output),
            "cleanup-owner-loss",
            str(tmp_path / "driver-status.json"),
        ],
        cwd=root,
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    state = _read(tmp_path / "actual-monitor-state.json")
    record = _read(output / "original-seven-duration-samples.json")
    _assert_samples(record)
    assert state["changed_after_full_command"] is True
    assert state["audit_interference"] is True
    assert state["owner_after_command"] is None
    assert record["raw_retention_valid"] is False and record["performance_admissible"] is False
    errors = record["observer_errors"]
    assert isinstance(errors, list) and any(
        "monitor cleanup failed" in str(error) for error in errors
    )
