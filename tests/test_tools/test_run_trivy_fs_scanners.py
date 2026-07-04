# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Trivy Filesystem Scanner Runner Tests

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = REPO_ROOT / "tools" / "security_scan" / "run_trivy_fs_scanners.py"


def _load_tool() -> Any:
    spec = importlib.util.spec_from_file_location("run_trivy_fs_scanners", TOOL_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_trivy_fs_scanner_blocks_high_critical_vulnerabilities(tmp_path: Path) -> None:
    tool = _load_tool()
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        output_path = Path(command[command.index("--output") + 1])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps({"Results": [{"Target": "repo", "Vulnerabilities": []}]}) + "\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, stdout="clean\n", stderr="")

    summary = tool.run_trivy_fs_scanner(
        repo_root=REPO_ROOT,
        output_dir=tmp_path,
        run_command=fake_run,
    )

    assert calls == [
        [
            "trivy",
            "fs",
            "--format",
            "json",
            "--output",
            str(tmp_path / "security" / "trivy_fs.json"),
            "--exit-code",
            "1",
            "--severity",
            "HIGH,CRITICAL",
            "--ignore-unfixed",
            "--scanners",
            "vuln",
            str(REPO_ROOT),
        ]
    ]
    assert summary["schema_version"] == tool.TRIVY_FS_SCANNER_SCHEMA_VERSION
    assert summary["passed"] is True
    assert summary["vulnerability_count"] == 0
    written = json.loads((tmp_path / "security" / "trivy_fs_summary.json").read_text())
    assert written == summary


def test_trivy_fs_scanner_fails_when_vulnerabilities_are_reported(tmp_path: Path) -> None:
    tool = _load_tool()

    def fake_run(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        output_path = Path(command[command.index("--output") + 1])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {
                    "Results": [
                        {
                            "Target": "repo",
                            "Vulnerabilities": [{"VulnerabilityID": "CVE-2099-0001"}],
                        }
                    ]
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="vulnerability found\n")

    summary = tool.run_trivy_fs_scanner(
        repo_root=REPO_ROOT,
        output_dir=tmp_path,
        run_command=fake_run,
    )

    assert summary["passed"] is False
    assert summary["vulnerability_count"] == 1
    assert summary["vulnerability_ids"] == ["CVE-2099-0001"]
    assert summary["returncode"] == 1


def test_trivy_fs_scanner_cli_returns_nonzero_for_failed_scan(tmp_path: Path) -> None:
    tool = _load_tool()

    def fake_runner(**_kwargs: Any) -> dict[str, Any]:
        return {
            "schema_version": tool.TRIVY_FS_SCANNER_SCHEMA_VERSION,
            "passed": False,
            "vulnerability_count": 1,
        }

    assert (
        tool.main(
            ["--repo-root", str(REPO_ROOT), "--output-dir", str(tmp_path)], runner=fake_runner
        )
        == 1
    )
