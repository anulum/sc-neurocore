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


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_tool() -> Any:
    repo_root = _repo_root()
    tool_path = repo_root / "tools" / "security_scan" / "run_syft_cyclonedx_scanners.py"
    spec = importlib.util.spec_from_file_location("run_syft_cyclonedx_scanners", tool_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_manifest_tool() -> Any:
    repo_root = _repo_root()
    tool_path = repo_root / "tools" / "security_scanner_manifest.py"
    spec = importlib.util.spec_from_file_location("security_scanner_manifest", tool_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_manifest_syft_command_uses_packet_runner() -> None:
    manifest_tool = _load_manifest_tool()
    manifest = manifest_tool.build_scanner_manifest()
    scanners = {scanner["name"]: scanner for scanner in manifest["scanners"]}

    assert scanners["syft-cyclonedx"]["command"] == (
        "python tools/security_scan/run_syft_cyclonedx_scanners.py "
        "--output-dir security/ci-security-packet"
    )
    assert scanners["syft-cyclonedx"]["pinned_version"] == "syft==1.20.0"


def test_runner_writes_and_validates_cyclonedx_sbom(tmp_path: Path) -> None:
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
        output_spec = command[command.index("--output") + 1]
        output_path = Path(output_spec.split("=", maxsplit=1)[1])
        output_path.write_text(
            json.dumps(
                {
                    "bomFormat": "CycloneDX",
                    "specVersion": "1.6",
                    "components": [{"name": "sc-neurocore", "type": "library"}],
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    summary = tool.run_syft_cyclonedx_scanner(
        repo_root=_repo_root(),
        output_dir=tmp_path / "packet",
        run_command=fake_run,
    )

    assert calls == [
        [
            "syft",
            ".",
            "--source-name",
            "sc-neurocore",
            "--source-version",
            "3.15.1",
            "--output",
            f"cyclonedx-json={tmp_path / 'packet' / 'security' / 'sbom.cdx.json'}",
        ]
    ]
    assert summary["passed"] is True
    assert summary["component_count"] == 1
    assert (
        json.loads(
            (tmp_path / "packet" / "security" / "syft_cyclonedx_summary.json").read_text(
                encoding="utf-8"
            )
        )["passed"]
        is True
    )


def test_runner_fails_when_syft_does_not_write_sbom(tmp_path: Path) -> None:
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

    summary = tool.run_syft_cyclonedx_scanner(
        repo_root=_repo_root(),
        output_dir=tmp_path / "packet",
        run_command=fake_run,
    )

    assert summary["passed"] is False
    assert summary["validation_errors"] == ["missing SBOM artifact"]


def test_runner_fails_on_non_cyclonedx_payload(tmp_path: Path) -> None:
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
        output_spec = command[command.index("--output") + 1]
        output_path = Path(output_spec.split("=", maxsplit=1)[1])
        output_path.write_text('{"bomFormat":"other"}', encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    summary = tool.run_syft_cyclonedx_scanner(
        repo_root=_repo_root(),
        output_dir=tmp_path / "packet",
        run_command=fake_run,
    )

    assert summary["passed"] is False
    assert "bomFormat must be CycloneDX" in summary["validation_errors"]
