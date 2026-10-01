# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


def test_real_missing_repository_records_reuse_execution_error(tmp_path: Path) -> None:
    """A real missing cwd is recorded and cannot reuse a stale compliance result."""
    tool = _load_tool()
    packet = tmp_path / "packet"
    (packet / "security").mkdir(parents=True)
    (packet / "security/reuse.json").write_text('{"non_compliant":false}')
    summary = tool.run_python_compliance_scanners(
        repo_root=tmp_path / "missing-repository", output_dir=packet
    )
    assert summary["passed"] is False
    assert summary["failed_scanners"] == ["pip-audit"]
    assert summary["non_blocking_failed_scanners"] == ["reuse"]
    assert json.loads((packet / "security/reuse.json").read_text()) == {
        "execution_error": "FileNotFoundError"
    }


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
        "python tools/security_scan/python_dependency_audit.py "
        "--output-dir security/ci-security-packet"
    )

    assert scanners["reuse"]["pinned_version"] == "reuse==6.2.0"
    assert scanners["reuse"]["command"] == "reuse --root . lint --json"
    assert scanners["reuse"]["blocking_policy"] == "allowed_to_fail"
    assert isinstance(scanners["reuse"]["allowed_to_fail_rationale"], str)


def test_runner_retains_inventory_failure_and_real_reuse_output(tmp_path: Path) -> None:
    """Exercise the real runner; an empty repository has no auditable lock set."""
    tool = _load_tool()
    summary = tool.run_python_compliance_scanners(
        repo_root=tmp_path,
        output_dir=tmp_path / "packet",
    )
    assert summary["passed"] is False
    assert summary["failed_scanners"] == ["pip-audit"]
    security = tmp_path / "packet" / "security"
    audit = json.loads((security / "pip_audit.json").read_text())
    assert audit["coverage_complete"] is False and audit["profiles"] == []
    assert audit["errors"]
    assert isinstance(json.loads((security / "reuse.json").read_text()), dict)
    assert json.loads((security / "python_compliance_summary.json").read_text()) == summary


def test_runner_overwrites_stale_dependency_report_on_invalid_inventory(tmp_path: Path) -> None:
    """A prior green report cannot satisfy a later failed audit."""
    tool = _load_tool()
    security = tmp_path / "packet" / "security"
    security.mkdir(parents=True)
    (security / "pip_audit.json").write_text('{"passed":true,"coverage_complete":true}')
    summary = tool.run_python_compliance_scanners(
        repo_root=tmp_path, output_dir=tmp_path / "packet"
    )
    assert summary["passed"] is False
    assert "pip-audit" in summary["failed_scanners"]
    assert json.loads((security / "pip_audit.json").read_text())["passed"] is False


def test_real_reuse_failure_retains_its_non_blocking_policy(tmp_path: Path) -> None:
    """Missing REUSE or an unlicensed empty root never becomes a blocking scanner."""
    tool = _load_tool()
    summary = tool.run_python_compliance_scanners(
        repo_root=tmp_path, output_dir=tmp_path / "packet"
    )
    reuse = next(s for s in summary["scanners"] if s["name"] == "reuse")
    assert reuse["returncode"] != 0
    assert summary["non_blocking_failed_scanners"] == ["reuse"]
    assert "reuse" not in summary["failed_scanners"]


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
