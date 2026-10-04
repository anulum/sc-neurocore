# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — actual preflight Cargo prerequisite contracts

"""Exercise individual preflight gates through real isolated Python and Cargo processes."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
_CHILD = """
import json
import pathlib
import sys
sys.path.insert(0, sys.argv[1])
from tools.preflight import run_gate
command = json.loads(sys.argv[2])
first = run_gate(sys.argv[4], command)
if sys.argv[3] == "remove-engine":
    pathlib.Path("engine").rmdir()
    second = run_gate("cargo-native-requirements", command)
    sys.exit(0 if first and not second else 1)
sys.exit(0 if first else 1)
"""


def _run_cargo_gate(
    working_directory: Path,
    *,
    search_path: str | None = None,
    reject_command: bool = False,
    remove_engine_after_pass: bool = False,
    missing_command: bool = False,
    gate_name: str = "cargo-native-requirements",
) -> subprocess.CompletedProcess[str]:
    """Invoke the public single-gate API without executing the preflight suite."""
    env = os.environ.copy()
    if search_path is not None:
        env["PATH"] = search_path
    command = ["cargo", "--not-a-real-option" if reject_command else "--version"]
    return subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            _CHILD,
            str(REPO_ROOT),
            json.dumps(None if missing_command else command),
            "remove-engine" if remove_engine_after_pass else "single-gate",
            gate_name,
        ],
        cwd=working_directory,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )


def _require_native_cargo() -> None:
    """Reject an unavailable real Cargo prerequisite instead of skipping validation."""
    assert shutil.which("cargo") is not None, "Real Cargo is required for native gate validation"


def test_missing_cargo_fails_with_existing_engine(tmp_path: Path) -> None:
    """An actually empty executable search path must refuse the Cargo gate."""
    (tmp_path / "engine").mkdir()
    result = _run_cargo_gate(tmp_path, search_path="")
    assert result.returncode == 1, result.stdout + result.stderr
    assert "FAIL:" in result.stdout
    assert "SKIP:" not in result.stdout


@pytest.mark.parametrize("engine_is_file", [False, True])
def test_missing_engine_directory_fails(tmp_path: Path, engine_is_file: bool) -> None:
    """Real Cargo availability does not admit an absent directory or a regular file."""
    _require_native_cargo()
    if engine_is_file:
        (tmp_path / "engine").write_text("not a directory", encoding="utf-8")
    result = _run_cargo_gate(tmp_path)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "FAIL:" in result.stdout
    assert "SKIP:" not in result.stdout


def test_available_cargo_executes_the_native_command(tmp_path: Path) -> None:
    """A valid prerequisite permits actual command execution and forwards success."""
    _require_native_cargo()
    (tmp_path / "engine").mkdir()
    result = _run_cargo_gate(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS: cargo-native-requirements" in result.stdout
    assert "cargo " in result.stdout


def test_native_cargo_failure_is_forwarded(tmp_path: Path) -> None:
    """An actual invalid Cargo option returns failure after prerequisites pass."""
    _require_native_cargo()
    (tmp_path / "engine").mkdir()
    result = _run_cargo_gate(tmp_path, reject_command=True)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "FAIL: cargo-native-requirements" in result.stdout
    assert "--not-a-real-option" in result.stderr


def test_engine_removal_invalidates_a_previous_success(tmp_path: Path) -> None:
    """Two real API calls must recheck prerequisites after the engine disappears."""
    _require_native_cargo()
    (tmp_path / "engine").mkdir()
    result = _run_cargo_gate(tmp_path, remove_engine_after_pass=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("PASS: cargo-native-requirements") == 1
    assert result.stdout.count("FAIL: cargo-native-requirements") == 1


@pytest.mark.parametrize("gate_name", ["cargo-native-requirements", "unconfigured-check"])
def test_absent_external_command_is_refused(tmp_path: Path, gate_name: str) -> None:
    """An absent external command cannot pass a Cargo gate or a custom gate label."""
    _require_native_cargo()
    (tmp_path / "engine").mkdir()
    result = _run_cargo_gate(tmp_path, missing_command=True, gate_name=gate_name)
    assert result.returncode == 1, result.stdout + result.stderr
    assert f"FAIL: {gate_name}" in result.stdout
