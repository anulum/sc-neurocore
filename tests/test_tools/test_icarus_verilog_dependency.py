# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Icarus Verilog dependency contract tests

"""Tests for the Icarus Verilog co-simulation dependency contract."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import pytest
import yaml

from tools import check_icarus_verilog

REPO_ROOT = Path(__file__).resolve().parents[2]


def _completed(command: Sequence[str], output: str, returncode: int = 0) -> subprocess.CompletedProcess[str]:
    """Return a typed completed process for fake version commands."""
    return subprocess.CompletedProcess(
        args=list(command),
        returncode=returncode,
        stdout=output,
        stderr="",
    )


def _load_ci_workflow() -> dict[str, Any]:
    """Load the live CI workflow from the repository."""
    workflow_path = REPO_ROOT / ".github" / "workflows" / "ci.yml"
    assert workflow_path.exists()
    workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    assert isinstance(workflow, dict)
    return cast(dict[str, Any], workflow)


def test_parse_iverilog_major_reads_stable_version() -> None:
    """The checker accepts the version banner emitted by Icarus Verilog 12."""
    assert (
        check_icarus_verilog.parse_iverilog_major(
            "Icarus Verilog version 12.0 (stable) ()"
        )
        == 12
    )


def test_parse_iverilog_major_returns_none_without_banner() -> None:
    """Unexpected command output does not produce a guessed version."""
    assert check_icarus_verilog.parse_iverilog_major("not an iverilog banner") is None


def test_run_command_captures_process_output() -> None:
    """The command runner captures text output from a version command."""
    result = check_icarus_verilog._run_command((sys.executable, "--version"))

    assert result.returncode == 0
    assert "Python" in result.stdout


def test_check_icarus_verilog_accepts_ci_floor() -> None:
    """The dependency check accepts Icarus Verilog 12 plus a working ``vvp``."""

    def runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        if command[0] == "iverilog":
            return _completed(command, "Icarus Verilog version 12.0 (stable) ()")
        return _completed(command, "Icarus Verilog VVP Code Generator 12.0")

    assert check_icarus_verilog.check_icarus_verilog(
        minimum_major=12,
        runner=runner,
    ) == []


def test_check_icarus_verilog_requires_frontend() -> None:
    """The dependency check fails closed when ``iverilog`` is missing."""

    def runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError(command[0])

    assert check_icarus_verilog.check_icarus_verilog(
        minimum_major=12,
        runner=runner,
    ) == ["iverilog executable is not available on PATH"]


def test_check_icarus_verilog_reports_bad_frontend_output() -> None:
    """The dependency check reports failed or unparsable frontend output."""

    def runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        if command[0] == "iverilog":
            return _completed(command, "unexpected", returncode=2)
        return _completed(command, "Icarus Verilog VVP Code Generator 12.0")

    errors = check_icarus_verilog.check_icarus_verilog(
        minimum_major=12,
        runner=runner,
    )

    assert "iverilog -V failed with exit code 2" in errors
    assert "iverilog -V output did not include an Icarus Verilog version" in errors


def test_check_icarus_verilog_rejects_old_major() -> None:
    """The dependency check rejects an older Icarus Verilog major version."""

    def runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        if command[0] == "iverilog":
            return _completed(command, "Icarus Verilog version 11.0 (stable) ()")
        return _completed(command, "Icarus Verilog VVP Code Generator 11.0")

    errors = check_icarus_verilog.check_icarus_verilog(
        minimum_major=12,
        runner=runner,
    )

    assert "below the required 12.x floor" in errors[0]


def test_check_icarus_verilog_reports_vvp_failure() -> None:
    """The dependency check reports a failing ``vvp`` version command."""

    def runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        if command[0] == "iverilog":
            return _completed(command, "Icarus Verilog version 12.0 (stable) ()")
        return _completed(command, "vvp failed", returncode=3)

    assert check_icarus_verilog.check_icarus_verilog(
        minimum_major=12,
        runner=runner,
    ) == ["vvp -V failed with exit code 3"]


def test_check_icarus_verilog_requires_vvp() -> None:
    """The dependency check fails closed when ``vvp`` is missing."""

    def runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        if command[0] == "iverilog":
            return _completed(command, "Icarus Verilog version 12.0 (stable) ()")
        raise FileNotFoundError(command[0])

    assert check_icarus_verilog.check_icarus_verilog(
        minimum_major=12,
        runner=runner,
    ) == ["vvp executable is not available on PATH"]


def test_main_reports_success(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The CLI returns success when the dependency check has no errors."""

    def fake_check(*, minimum_major: int) -> list[str]:
        assert minimum_major == 12
        return []

    monkeypatch.setattr(check_icarus_verilog, "check_icarus_verilog", fake_check)

    assert check_icarus_verilog.main(["--minimum-major", "12"]) == 0
    assert "[OK] Icarus Verilog toolchain satisfies 12.x floor" in capsys.readouterr().out


def test_main_reports_failures(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The CLI returns non-zero and prints dependency errors."""

    def fake_check(*, minimum_major: int) -> list[str]:
        assert minimum_major == 12
        return ["missing simulator"]

    monkeypatch.setattr(check_icarus_verilog, "check_icarus_verilog", fake_check)

    assert check_icarus_verilog.main(["--minimum-major", "12"]) == 1
    assert "[FAIL] missing simulator" in capsys.readouterr().out


def test_ci_workflow_checks_icarus_verilog_after_install() -> None:
    """The live CI workflow installs and verifies Icarus before package tests."""
    workflow = _load_ci_workflow()
    steps = workflow["jobs"]["test"]["steps"]
    step_names = [step.get("name", "") for step in steps]

    install_index = step_names.index("Install HDL simulator")
    verify_index = step_names.index("Verify HDL simulator versions")
    package_index = step_names.index("Install package")

    install_run = steps[install_index]["run"]
    verify_run = steps[verify_index]["run"]
    assert "apt-get install -y -qq cvc5 git iverilog make verilator yosys" in install_run
    assert "python tools/check_icarus_verilog.py --minimum-major 12" in verify_run
    assert install_index < verify_index < package_index


def test_fpga_toolchain_guide_documents_ci_floor() -> None:
    """The hardware guide records the Icarus version and CI check contract."""
    guide = (REPO_ROOT / "docs" / "hardware" / "FPGA_TOOLCHAIN_GUIDE.md").read_text(
        encoding="utf-8"
    )

    assert "Icarus Verilog 12.0 or newer" in guide
    assert "python tools/check_icarus_verilog.py --minimum-major 12" in guide
    assert "`iverilog -V`" in guide
    assert "`vvp -V`" in guide
