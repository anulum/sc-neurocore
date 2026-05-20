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
    tool_path = repo_root / "tools" / "security_scan" / "run_rust_security_scanners.py"
    spec = importlib.util.spec_from_file_location("run_rust_security_scanners", tool_path)
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


def test_manifest_rust_commands_use_stdout_reports_not_lockfile_output() -> None:
    manifest_tool = _load_manifest_tool()
    manifest = manifest_tool.build_scanner_manifest()
    commands = {scanner["name"]: scanner["command"] for scanner in manifest["scanners"]}

    assert commands["cargo-audit"] == "cargo audit --format json --file Cargo.lock"
    assert "security/cargo_audit.json" not in commands["cargo-audit"]

    assert commands["cargo-deny"].startswith(
        "cargo deny --format json --manifest-path engine/Cargo.toml check"
    )
    assert "--config engine/deny.toml" in commands["cargo-deny"]
    assert commands["cargo-deny"].endswith("licenses")


def test_runner_captures_cargo_stdout_to_security_artifacts(tmp_path: Path) -> None:
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
        return subprocess.CompletedProcess(command, 0, stdout='{"ok":true}', stderr="")

    summary = tool.run_rust_scanners(
        repo_root=tmp_path,
        output_dir=tmp_path / "packet",
        run_command=fake_run,
    )

    assert [call[:2] for call in calls] == [["cargo", "audit"], ["cargo", "deny"]]
    assert summary["passed"] is True
    assert summary["failed_scanners"] == []
    assert json.loads(
        (tmp_path / "packet" / "security" / "cargo_audit.json").read_text(encoding="utf-8")
    ) == {"ok": True}
    assert json.loads(
        (tmp_path / "packet" / "security" / "cargo_deny.json").read_text(encoding="utf-8")
    ) == {"ok": True}


def test_runner_records_invalid_json_as_raw_output(tmp_path: Path) -> None:
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
        return subprocess.CompletedProcess(command, 1, stdout="not-json", stderr="failed")

    summary = tool.run_rust_scanners(
        repo_root=tmp_path,
        output_dir=tmp_path / "packet",
        run_command=fake_run,
    )

    assert summary["passed"] is False
    assert summary["failed_scanners"] == ["cargo-audit", "cargo-deny"]
    assert json.loads(
        (tmp_path / "packet" / "security" / "cargo_audit.json").read_text(encoding="utf-8")
    ) == {"raw_stdout": "not-json"}


def test_runner_captures_json_lines_from_stderr_when_stdout_is_empty(tmp_path: Path) -> None:
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
        if command[:2] == ["cargo", "deny"]:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout="",
                stderr='{"type":"diagnostic"}\n{"type":"summary"}\n',
            )
        return subprocess.CompletedProcess(command, 0, stdout='{"ok":true}', stderr="")

    tool.run_rust_scanners(
        repo_root=tmp_path,
        output_dir=tmp_path / "packet",
        run_command=fake_run,
    )

    assert json.loads(
        (tmp_path / "packet" / "security" / "cargo_deny.json").read_text(encoding="utf-8")
    ) == [{"type": "diagnostic"}, {"type": "summary"}]
