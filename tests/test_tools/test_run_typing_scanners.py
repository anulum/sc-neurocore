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
    tool_path = repo_root / "tools" / "security_scan" / "run_typing_scanners.py"
    spec = importlib.util.spec_from_file_location("run_typing_scanners", tool_path)
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


def test_manifest_typing_commands_are_executable_and_pinned() -> None:
    manifest_tool = _load_manifest_tool()
    manifest = manifest_tool.build_scanner_manifest()
    scanners = {scanner["name"]: scanner for scanner in manifest["scanners"]}

    assert scanners["pyright"]["pinned_version"] == "pyright==1.1.382"
    assert scanners["pyright"]["command"] == "pyright --project pyrightconfig.json --outputjson"

    assert scanners["mypy"]["pinned_version"] == "mypy==1.15.0"
    assert scanners["mypy"]["command"] == "mypy --strict --json-report security/mypy ."


def test_runner_writes_pyright_json_and_mypy_report_summary(tmp_path: Path) -> None:
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
        if Path(command[0]).name == "pyright":
            return subprocess.CompletedProcess(
                command,
                0,
                stdout='{"summary":{"errorCount":0}}',
                stderr="",
            )
        report_dir = Path(command[command.index("--json-report") + 1])
        report_dir.mkdir(parents=True)
        (report_dir / "index.json").write_text('{"summary":{"files":1}}', encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="Success", stderr="")

    summary = tool.run_typing_scanners(
        repo_root=tmp_path,
        output_dir=tmp_path / "packet",
        run_command=fake_run,
    )

    assert [Path(call[0]).name for call in calls] == ["pyright", "mypy"]
    assert summary["passed"] is True
    assert summary["failed_scanners"] == []
    assert json.loads(
        (tmp_path / "packet" / "security" / "pyright.json").read_text(encoding="utf-8")
    ) == {"summary": {"errorCount": 0}}
    assert (tmp_path / "packet" / "security" / "mypy" / "index.json").exists()
    assert (tmp_path / "packet" / "security" / "typing_scanner_summary.json").exists()


def test_runner_records_invalid_pyright_json_as_raw_output(tmp_path: Path) -> None:
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
        if Path(command[0]).name == "pyright":
            return subprocess.CompletedProcess(command, 1, stdout="not-json", stderr="failed")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    summary = tool.run_typing_scanners(
        repo_root=tmp_path,
        output_dir=tmp_path / "packet",
        run_command=fake_run,
    )

    assert summary["passed"] is False
    assert "pyright" in summary["failed_scanners"]
    assert json.loads(
        (tmp_path / "packet" / "security" / "pyright.json").read_text(encoding="utf-8")
    ) == {"raw_stdout": "not-json", "raw_stderr": "failed"}


def test_runner_resolves_tools_next_to_active_python(tmp_path: Path, monkeypatch: Any) -> None:
    tool = _load_tool()
    fake_python = tmp_path / "venv" / "bin" / "python"
    fake_tool = tmp_path / "venv" / "bin" / "pyright"
    fake_tool.parent.mkdir(parents=True)
    fake_tool.write_text("#!/bin/sh\n", encoding="utf-8")
    fake_tool.chmod(0o755)
    monkeypatch.setattr(tool.sys, "executable", str(fake_python))
    monkeypatch.setattr(tool.shutil, "which", lambda _name: None)

    resolved = tool._resolve_tool("pyright")

    assert resolved == str(fake_tool)
