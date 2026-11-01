# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Python compliance scanner lane

"""The Python compliance lane blocks on any pip-audit or REUSE failure, using real scanner runs."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest


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
    assert summary["failed_scanners"] == ["pip-audit", "reuse"]
    assert summary["non_blocking_failed_scanners"] == []
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
    """Both compliance scanners are pinned, and REUSE is recorded as blocking."""
    manifest_tool = _load_manifest_tool()
    manifest = manifest_tool.build_scanner_manifest()
    scanners = {scanner["name"]: scanner for scanner in manifest["scanners"]}

    assert scanners["pip-audit"]["pinned_version"] == "pip-audit==2.10.1"
    assert scanners["pip-audit"]["command"] == (
        "python tools/security_scan/python_dependency_audit.py "
        "--output-dir security/ci-security-packet"
    )

    assert scanners["reuse"]["pinned_version"] == "reuse==6.3.0"
    assert scanners["reuse"]["command"] == "reuse --root . lint --json"
    assert scanners["reuse"]["blocking_policy"] == "blocking"
    assert scanners["reuse"]["allowed_to_fail_rationale"] is None
    assert not _load_tool().NON_BLOCKING_SCANNERS


def test_runner_retains_inventory_failure_and_real_reuse_output(tmp_path: Path) -> None:
    """Exercise the real runner; an empty repository has no auditable lock set."""
    tool = _load_tool()
    summary = tool.run_python_compliance_scanners(
        repo_root=tmp_path,
        output_dir=tmp_path / "packet",
    )
    assert summary["passed"] is False
    assert summary["failed_scanners"] == ["pip-audit", "reuse"]
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


def test_real_reuse_failure_blocks_the_lane(tmp_path: Path) -> None:
    """A REUSE failure on an unlicensed empty root fails the lane; nothing is non-blocking."""
    tool = _load_tool()
    summary = tool.run_python_compliance_scanners(
        repo_root=tmp_path, output_dir=tmp_path / "packet"
    )
    reuse = next(s for s in summary["scanners"] if s["name"] == "reuse")
    assert reuse["returncode"] != 0
    assert summary["non_blocking_failed_scanners"] == []
    assert "reuse" in summary["failed_scanners"]
    assert summary["passed"] is False


def test_runner_resolves_tools_next_to_active_python(tmp_path: Path, monkeypatch: Any) -> None:
    """A scanner installed next to the active interpreter is used when PATH has none."""
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


def test_a_file_without_licensing_information_fails_the_lane(tmp_path: Path) -> None:
    """REUSE blocks: a real repository with one unlicensed file fails the compliance lane."""
    tool = _load_tool()
    repo = tmp_path / "repository"
    repo.mkdir()
    (repo / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    packet = tmp_path / "packet"

    summary = tool.run_python_compliance_scanners(repo_root=repo, output_dir=packet)

    assert summary["passed"] is False
    assert "reuse" in summary["failed_scanners"]
    assert summary["non_blocking_failed_scanners"] == []
    report = json.loads((packet / "security/reuse.json").read_text())
    assert "module.py" in json.dumps(report)


@pytest.mark.parametrize(("passed", "status"), [(True, 0), (False, 1)])
def test_command_line_status_follows_the_compliance_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], passed: bool, status: int
) -> None:
    """The exit status is 0 only for a passing summary, which is printed in full."""
    tool = _load_tool()
    received: dict[str, Any] = {}

    def runner(**kwargs: Any) -> dict[str, Any]:
        received.update(kwargs)
        return {"passed": passed, "failed_scanners": [] if passed else ["reuse"]}

    assert tool.main(["--output-dir", str(tmp_path)], runner=runner) == status
    assert json.loads(capsys.readouterr().out)["passed"] is passed
    assert received == {"repo_root": tool._project_root(), "output_dir": tmp_path}
