# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Gitleaks Scanner Runner Tests

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = REPO_ROOT / "tools" / "security_scan" / "run_gitleaks_scanners.py"


def _load_tool() -> Any:
    spec = importlib.util.spec_from_file_location("run_gitleaks_scanners", TOOL_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_gitleaks_scanner_writes_low_noise_release_artifacts(tmp_path: Path) -> None:
    tool = _load_tool()
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        output_path = Path(command[command.index("--report-path") + 1])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("[]\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="clean\n", stderr="")

    summary = tool.run_gitleaks_scanner(
        repo_root=REPO_ROOT,
        output_dir=tmp_path,
        run_command=fake_run,
    )

    assert calls == [
        [
            "gitleaks",
            "detect",
            "--source",
            str(REPO_ROOT),
            "--report-format",
            "json",
            "--report-path",
            str(tmp_path / "security" / "gitleaks.json"),
            "--no-banner",
            "--redact",
        ]
    ]
    assert summary["schema_version"] == tool.GITLEAKS_SCANNER_SCHEMA_VERSION
    assert summary["passed"] is True
    assert summary["finding_count"] == 0
    assert summary["allowed_to_fail"] is True
    written = json.loads((tmp_path / "security" / "gitleaks_summary.json").read_text())
    assert written == summary


def test_gitleaks_scanner_records_findings_without_blocking_packet(tmp_path: Path) -> None:
    tool = _load_tool()

    def fake_run(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        output_path = Path(command[command.index("--report-path") + 1])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps([{"RuleID": "generic-api-key"}]) + "\n")
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="1 leak found\n")

    summary = tool.run_gitleaks_scanner(
        repo_root=REPO_ROOT,
        output_dir=tmp_path,
        run_command=fake_run,
    )

    assert summary["passed"] is True
    assert summary["finding_count"] == 1
    assert summary["leak_detected"] is True
    assert summary["returncode"] == 1


def test_gitleaks_scanner_cli_returns_nonzero_for_execution_failure(tmp_path: Path) -> None:
    tool = _load_tool()

    def fake_runner(**_kwargs: Any) -> dict[str, Any]:
        return {
            "schema_version": tool.GITLEAKS_SCANNER_SCHEMA_VERSION,
            "passed": False,
            "finding_count": 0,
            "validation_errors": ["missing Gitleaks report artifact"],
        }

    assert (
        tool.main(
            ["--repo-root", str(REPO_ROOT), "--output-dir", str(tmp_path)], runner=fake_runner
        )
        == 1
    )
