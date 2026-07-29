# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def _load_tool() -> Any:
    repo_root = Path(__file__).resolve().parents[2]
    tool_path = repo_root / "tools" / "security_scan" / "run_python_compliance_scanners.py"
    spec = importlib.util.spec_from_file_location("run_python_compliance_scanners", tool_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_manifest_tool() -> Any:
    repo_root = Path(__file__).resolve().parents[2]
    tool_path = repo_root / "tools" / "security_scanner_manifest.py"
    spec = importlib.util.spec_from_file_location("security_scanner_manifest", tool_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_manifest_python_compliance_commands_are_executable_and_pinned() -> None:
    manifest_tool = _load_manifest_tool()
    manifest = manifest_tool.build_scanner_manifest()
    scanners = {scanner["name"]: scanner for scanner in manifest["scanners"]}

    assert scanners["pip-audit"]["pinned_version"] == "pip-audit==2.10.1"
    assert scanners["pip-audit"]["command"] == (
        "pip-audit --strict --requirement requirements/release.txt "
        "--format json --progress-spinner off --output security/pip_audit.json"
    )

    assert scanners["reuse"]["pinned_version"] == "reuse==6.2.0"
    assert scanners["reuse"]["command"] == "reuse --root . lint --json"
    assert scanners["reuse"]["blocking_policy"] == "allowed_to_fail"
    assert isinstance(scanners["reuse"]["allowed_to_fail_rationale"], str)


def test_runner_captures_pip_audit_output_file_and_reuse_stdout(tmp_path: Path) -> None:
    tool = _load_tool()
    calls: list[list[str]] = []

    def fake_run(
        command: list[str],
        *,
        cwd: Path,
        capture_output: bool,
        text: bool,
        timeout: int,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        del cwd, capture_output, text, timeout, check
        calls.append(command)
        if Path(command[0]).name == "pip-audit":
            Path(command[-1]).write_text('{"dependencies":[]}', encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(command, 0, stdout='{"summary":{}}', stderr="")

    summary = tool.run_python_compliance_scanners(
        repo_root=tmp_path,
        output_dir=tmp_path / "packet",
        run_command=fake_run,
    )

    assert [Path(call[0]).name for call in calls] == ["pip-audit", "reuse"]
    assert summary["passed"] is True
    assert summary["failed_scanners"] == []
    assert json.loads(
        (tmp_path / "packet" / "security" / "pip_audit.json").read_text(encoding="utf-8")
    ) == {"dependencies": []}
    assert json.loads(
        (tmp_path / "packet" / "security" / "reuse.json").read_text(encoding="utf-8")
    ) == {"summary": {}}


def test_runner_fails_when_pip_audit_does_not_write_report(tmp_path: Path) -> None:
    tool = _load_tool()

    def fake_run(
        command: list[str],
        *,
        cwd: Path,
        capture_output: bool,
        text: bool,
        timeout: int,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        del cwd, capture_output, text, timeout, check
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    summary = tool.run_python_compliance_scanners(
        repo_root=tmp_path,
        output_dir=tmp_path / "packet",
        run_command=fake_run,
    )

    assert summary["passed"] is False
    assert "pip-audit" in summary["failed_scanners"]


def test_reuse_failure_is_reported_without_failing_python_compliance_lane(
    tmp_path: Path,
) -> None:
    tool = _load_tool()

    def fake_run(
        command: list[str],
        *,
        cwd: Path,
        capture_output: bool,
        text: bool,
        timeout: int,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        del cwd, capture_output, text, timeout, check
        if Path(command[0]).name == "pip-audit":
            Path(command[-1]).write_text('{"dependencies":[]}', encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(
            command, 1, stdout='{"summary":{"compliant":false}}', stderr=""
        )

    summary = tool.run_python_compliance_scanners(
        repo_root=tmp_path,
        output_dir=tmp_path / "packet",
        run_command=fake_run,
    )

    assert summary["passed"] is True
    assert summary["failed_scanners"] == []
    assert summary["non_blocking_failed_scanners"] == ["reuse"]


def test_runner_resolves_tools_next_to_active_python(tmp_path: Path, monkeypatch: Any) -> None:
    tool = _load_tool()
    fake_python = tmp_path / "venv" / "bin" / "python"
    fake_tool = tmp_path / "venv" / "bin" / "pip-audit"
    fake_tool.parent.mkdir(parents=True)
    fake_tool.write_text("#!/bin/sh\n", encoding="utf-8")
    fake_tool.chmod(0o755)
    monkeypatch.setattr(tool.sys, "executable", str(fake_python))
    monkeypatch.setattr(tool.shutil, "which", lambda _name: None)

    resolved = tool._resolve_tool("pip-audit")

    assert resolved == str(fake_tool)
