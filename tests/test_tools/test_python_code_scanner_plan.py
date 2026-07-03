# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest


def _load_tool() -> Any:
    repo_root = Path(__file__).resolve().parents[2]
    tool_path = repo_root / "tools" / "security_scan" / "python_code_scanner_plan.py"
    spec = importlib.util.spec_from_file_location("python_code_scanner_plan", tool_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


EXPECTED_SCANNERS = {"bandit", "mypy", "pip-audit", "pyright", "ruff", "semgrep"}


def _manifest_with_scanners(root: Path, *, missing_required: bool = False) -> dict[str, Any]:
    _ = root
    _ = missing_required
    return {
        "schema_version": "sc-neurocore.security-scanner-manifest.v1",
        "scanners": [
            {
                "name": "ruff",
                "command": "ruff check src",
                "inputs": [
                    {
                        "path": "src",
                        "purpose": "Python sources",
                        "required": True,
                    }
                ],
            },
            {
                "name": "pyright",
                "command": "pyright",
                "inputs": [
                    {
                        "path": "pyrightconfig.json",
                        "purpose": "pyright config",
                        "required": True,
                    }
                ],
            },
            {
                "name": "mypy",
                "command": "mypy .",
                "inputs": [
                    {
                        "path": "src",
                        "purpose": "typed source surface",
                        "required": True,
                    }
                ],
            },
            {
                "name": "semgrep",
                "command": (
                    "semgrep scan --config .semgrep.yml --json --error "
                    "--output security/semgrep.json src tools"
                ),
                "inputs": [
                    {
                        "path": ".semgrep.yml",
                        "purpose": "semgrep policy",
                        "required": True,
                    },
                    {
                        "path": "src",
                        "purpose": "python code",
                        "required": True,
                    },
                    {
                        "path": "tools",
                        "purpose": "tooling code",
                        "required": True,
                    },
                ],
            },
            {
                "name": "bandit",
                "command": "bandit -q -r src",
                "inputs": [
                    {
                        "path": "src",
                        "purpose": "python code",
                        "required": True,
                    }
                ],
            },
            {
                "name": "pip-audit",
                "command": "pip-audit",
                "inputs": [
                    {
                        "path": "requirements/release.txt",
                        "purpose": "requirements",
                        "required": True,
                    }
                ],
            },
        ],
    }


def test_python_code_plan_includes_expected_scanners_and_is_deterministic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tool = _load_tool()
    manifest = _manifest_with_scanners(tmp_path)
    monkeypatch.setattr(tool, "_load_manifest", lambda: manifest)
    monkeypatch.setattr(tool, "_is_tool_available", lambda _name: "/usr/bin/fake")

    def exists(path: Path) -> bool:
        return (
            not str(path).endswith("requirements/release.txt")
            or (tmp_path / "requirements" / "release.txt").exists()
        )

    (tmp_path / "requirements").mkdir()
    (tmp_path / "requirements" / "release.txt").write_text("", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "tools").mkdir()
    (tmp_path / ".semgrep.yml").write_text("rules: []\n", encoding="utf-8")
    (tmp_path / "pyrightconfig.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(tool, "_input_exists", exists)

    first = tool.build_scanner_plan(repo_root=tmp_path)
    second = tool.build_scanner_plan(repo_root=tmp_path)

    assert first == second
    assert first["schema_version"] == "sc-neurocore.python-code-scanner-plan.v1"
    assert first["scanner_count"] == len(first["scanners"])

    scanner_names = [scanner["name"] for scanner in first["scanners"]]
    assert set(scanner_names) == EXPECTED_SCANNERS
    assert scanner_names == sorted(scanner_names)


def test_plan_does_not_execute_scanner_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tool = _load_tool()
    manifest = _manifest_with_scanners(tmp_path)
    monkeypatch.setattr(tool, "_load_manifest", lambda: manifest)
    monkeypatch.setattr(tool, "_is_tool_available", lambda _name: "/usr/bin/fake")

    (tmp_path / "requirements").mkdir()
    (tmp_path / "requirements" / "release.txt").write_text("", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "tools").mkdir()
    (tmp_path / ".semgrep.yml").write_text("rules: []\n", encoding="utf-8")
    (tmp_path / "pyrightconfig.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(tool, "_input_exists", lambda _path: True)

    called = {"subprocess_run": 0, "subprocess_call": 0, "system": 0}

    def fail_run(*_args: object, **_kwargs: object) -> None:
        called["subprocess_run"] += 1
        raise AssertionError("Scanner command executed")

    def fail_call(*_args: object, **_kwargs: object) -> None:
        called["subprocess_call"] += 1
        raise AssertionError("Scanner command executed")

    monkeypatch.setattr(subprocess, "run", fail_run)
    monkeypatch.setattr(subprocess, "call", fail_call)
    monkeypatch.setattr(
        os,
        "system",
        lambda *args: (_ for _ in ()).throw(AssertionError("Scanner command executed")),
    )

    plan = tool.build_scanner_plan(repo_root=tmp_path)

    assert called == {"subprocess_run": 0, "subprocess_call": 0, "system": 0}
    assert set(scanner["name"] for scanner in plan["scanners"]) == EXPECTED_SCANNERS


def test_missing_required_input_marks_scanner_missing_required_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tool = _load_tool()
    manifest = _manifest_with_scanners(tmp_path, missing_required=True)
    monkeypatch.setattr(tool, "_load_manifest", lambda: manifest)
    monkeypatch.setattr(tool, "_is_tool_available", lambda _name: "/usr/bin/fake")

    def exists(path: Path) -> bool:
        return False

    monkeypatch.setattr(tool, "_input_exists", exists)

    plan = tool.build_scanner_plan(repo_root=tmp_path)
    by_name = {scanner["name"]: scanner["run_class"] for scanner in plan["scanners"]}

    assert by_name["pip-audit"] == "missing_required_input"


def test_missing_tool_marks_scanner_missing_tool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tool = _load_tool()
    manifest = _manifest_with_scanners(tmp_path)
    monkeypatch.setattr(tool, "_load_manifest", lambda: manifest)
    monkeypatch.setattr(tool, "_is_tool_available", lambda _name: None)
    monkeypatch.setattr(tool, "_input_exists", lambda _path: True)

    plan = tool.build_scanner_plan(repo_root=tmp_path)
    by_name = {scanner["name"]: scanner["run_class"] for scanner in plan["scanners"]}

    assert by_name["ruff"] == "missing_tool"


def test_missing_executable_marks_deferred_heavy_and_cli_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tool = _load_tool()
    manifest = _manifest_with_scanners(tmp_path)
    monkeypatch.setattr(tool, "_load_manifest", lambda: manifest)
    monkeypatch.setattr(tool, "_is_tool_available", lambda _name: None)
    monkeypatch.setattr(tool, "_input_exists", lambda _path: True)
    output = tmp_path / "plan.json"

    tool.DEFERRED_HEAVY_SCANNERS = {"ruff"}
    plan = tool.build_scanner_plan(repo_root=tmp_path)
    ruff_entry = next(scanner for scanner in plan["scanners"] if scanner["name"] == "ruff")
    assert ruff_entry["run_class"] == "deferred_heavy"

    rc = tool.main(["--output", str(output), "--fail-on-missing-required-inputs"])
    assert rc == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["scanners"] == sorted(payload["scanners"], key=lambda item: item["name"])


def test_cli_defaults_do_not_fail_on_missing_tools_but_flag_fails_on_missing_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tool = _load_tool()
    manifest = _manifest_with_scanners(tmp_path)
    monkeypatch.setattr(tool, "_load_manifest", lambda: manifest)
    monkeypatch.setattr(tool, "_is_tool_available", lambda _name: None)
    monkeypatch.setattr(tool, "_input_exists", lambda _path: True)

    output = tmp_path / "plan.json"
    assert tool.main(["--output", str(output)]) == 0

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert all(scanner["run_class"] == "missing_tool" for scanner in payload["scanners"])

    # now simulate missing required input with a stricter validator
    monkeypatch.setattr(tool, "_input_exists", lambda _path: False)
    output_with_missing = tmp_path / "plan-missing-required.json"
    assert (
        tool.main(["--output", str(output_with_missing), "--fail-on-missing-required-inputs"]) == 1
    )
