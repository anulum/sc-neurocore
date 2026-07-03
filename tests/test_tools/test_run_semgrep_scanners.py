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
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = REPO_ROOT / "tools" / "security_scan" / "run_semgrep_scanners.py"


def _load_tool() -> Any:
    spec = importlib.util.spec_from_file_location("run_semgrep_scanners", TOOL_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_semgrep_scanner_uses_repo_owned_policy_and_writes_summary(tmp_path: Path) -> None:
    tool = _load_tool()
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        output_path = Path(command[command.index("--output") + 1])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps({"results": []}) + "\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="clean\n", stderr="")

    summary = tool.run_semgrep_scanner(
        repo_root=REPO_ROOT,
        output_dir=tmp_path,
        run_command=fake_run,
    )

    assert calls == [
        [
            "semgrep",
            "scan",
            "--config",
            str(REPO_ROOT / ".semgrep.yml"),
            "--json",
            "--error",
            "--output",
            str(tmp_path / "security" / "semgrep.json"),
            "src",
            "tools",
        ]
    ]
    assert summary["schema_version"] == tool.SEMGREP_SCANNER_SCHEMA_VERSION
    assert summary["passed"] is True
    assert summary["finding_count"] == 0
    written = json.loads((tmp_path / "security" / "semgrep_summary.json").read_text())
    assert written == summary


def test_semgrep_scanner_fails_when_semgrep_reports_findings(tmp_path: Path) -> None:
    tool = _load_tool()

    def fake_run(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        output_path = Path(command[command.index("--output") + 1])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps({"results": [{"check_id": "x"}]}) + "\n")
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="1 finding\n")

    summary = tool.run_semgrep_scanner(
        repo_root=REPO_ROOT,
        output_dir=tmp_path,
        run_command=fake_run,
    )

    assert summary["passed"] is False
    assert summary["finding_count"] == 1
    assert summary["returncode"] == 1


def test_semgrep_scanner_cli_returns_nonzero_for_failed_run(tmp_path: Path) -> None:
    tool = _load_tool()

    def fake_runner(**_kwargs: Any) -> dict[str, Any]:
        return {
            "schema_version": tool.SEMGREP_SCANNER_SCHEMA_VERSION,
            "passed": False,
            "finding_count": 1,
        }

    assert (
        tool.main(
            ["--repo-root", str(REPO_ROOT), "--output-dir", str(tmp_path)], runner=fake_runner
        )
        == 1
    )
