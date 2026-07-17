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
    tool_path = repo_root / "tools" / "security_scan" / "run_lightweight_security_scanners.py"
    spec = importlib.util.spec_from_file_location("run_lightweight_security_scanners", tool_path)
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


def test_runner_executes_lightweight_scanners_and_writes_summary(
    tmp_path: Path,
) -> None:
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
        if command[0] == "actionlint":
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    summary = tool.run_lightweight_scanners(
        repo_root=tmp_path,
        output_dir=tmp_path / "packet",
        run_command=fake_run,
    )

    assert [call[0] for call in calls] == ["ruff", "bandit", "actionlint"]
    assert summary["passed"] is True
    assert summary["scanner_count"] == 3
    assert [scanner["name"] for scanner in summary["scanners"]] == [
        "ruff",
        "bandit",
        "actionlint",
    ]
    assert (tmp_path / "packet" / "security" / "lightweight_scanner_summary.json").exists()


def test_manifest_lightweight_commands_match_runner_contract() -> None:
    manifest_tool = _load_manifest_tool()
    manifest = manifest_tool.build_scanner_manifest()
    manifest_by_name = {scanner["name"]: scanner["command"] for scanner in manifest["scanners"]}

    assert manifest_by_name["ruff"].startswith("ruff check --output-format json")
    assert "security/ruff.json" in manifest_by_name["ruff"]
    assert "--cache-dir security/ruff-cache" in manifest_by_name["ruff"]
    assert "src tools tests" in manifest_by_name["ruff"]

    assert manifest_by_name["bandit"].startswith(
        "bandit -q -c pyproject.toml -r src/sc_neurocore tools"
    )
    assert "src/sc_neurocore/accel/mojo/.pixi" in manifest_by_name["bandit"]
    assert "security/bandit.json" in manifest_by_name["bandit"]

    assert manifest_by_name["actionlint"] == "actionlint -format '{{json .}}'"


def test_ruff_cache_is_written_under_packet_security_dir(tmp_path: Path) -> None:
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
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    tool.run_lightweight_scanners(
        repo_root=tmp_path,
        output_dir=tmp_path / "packet",
        run_command=fake_run,
    )

    ruff_command = calls[0]
    assert "--cache-dir" in ruff_command
    cache_dir = Path(ruff_command[ruff_command.index("--cache-dir") + 1])
    assert cache_dir == tmp_path / "packet" / "security" / "ruff-cache"


def test_runner_parses_actionlint_json_lines(tmp_path: Path) -> None:
    tool = _load_tool()
    actionlint_payload = {
        "message": "bad workflow",
        "filepath": ".github/workflows/ci.yml",
        "line": 12,
    }

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
        if command[0] == "actionlint":
            return subprocess.CompletedProcess(
                command,
                1,
                stdout=json.dumps(actionlint_payload) + "\n",
                stderr="",
            )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    summary = tool.run_lightweight_scanners(
        repo_root=tmp_path,
        output_dir=tmp_path / "packet",
        run_command=fake_run,
    )

    actionlint_report = json.loads(
        (tmp_path / "packet" / "security" / "actionlint.json").read_text(encoding="utf-8")
    )
    assert actionlint_report == [actionlint_payload]
    assert summary["passed"] is False
    assert summary["failed_scanners"] == ["actionlint"]


def test_cli_returns_nonzero_when_lightweight_scanner_fails(
    tmp_path: Path,
) -> None:
    tool = _load_tool()

    def fake_runner(*, repo_root: Path, output_dir: Path) -> dict[str, Any]:
        del repo_root, output_dir
        return {"passed": False}

    assert (
        tool.main(
            ["--repo-root", str(tmp_path), "--output-dir", str(tmp_path / "packet")],
            runner=fake_runner,
        )
        == 1
    )
