# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


REPO = Path(__file__).resolve().parents[2]
TOOL = REPO / "tools/eda_toolchain_versions.py"


def _load_tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("eda_toolchain_versions", TOOL)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _version_lookup(distribution: str) -> str:
    if distribution == "pynq":
        return "3.1.0"
    raise AssertionError(f"unexpected distribution lookup: {distribution}")


def test_collect_versions_records_fallback_command_and_environment() -> None:
    tool = _load_tool()

    def runner(command: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        executable = command[0]
        if executable == "openroad" and command[1] == "-version":
            return subprocess.CompletedProcess(command, 1, stdout="", stderr="bad option\n")
        if executable == "openroad" and command[1] == "--version":
            return subprocess.CompletedProcess(command, 0, stdout="OpenROAD v2.0-2026-04\n")
        if executable == "yosys":
            return subprocess.CompletedProcess(command, 0, stdout="Yosys 0.33\n")
        raise FileNotFoundError(executable)

    report = tool.collect_eda_toolchain_versions(
        runner=runner,
        version_lookup=_version_lookup,
        environ={
            "OPENROAD_IMAGE_DIGEST": "sha256:abc123",
            "PDK": "sky130A",
            "PDK_ROOT": "/opt/pdk",
        },
    )

    assert report["schema_version"] == tool.SCHEMA_VERSION
    assert report["tools"]["openroad"]["available"] is True
    assert report["tools"]["openroad"]["command"] == "openroad --version"
    assert report["tools"]["openroad"]["version"] == "OpenROAD v2.0-2026-04"
    assert report["tools"]["vivado"]["available"] is False
    assert report["tools"]["pynq"]["version"] == "3.1.0"
    assert report["environment"] == {
        "openroad_image_digest": "sha256:abc123",
        "pdk": "sky130A",
        "pdk_root_set": True,
    }


def test_check_expectations_reports_required_and_version_failures() -> None:
    tool = _load_tool()
    report = {
        "tools": {
            "openroad": {"available": True, "version": "OpenROAD v2.0-2026-04"},
            "vivado": {"available": False, "version": None},
        }
    }

    findings = tool.check_expectations(
        report,
        required=["vivado", "yosys"],
        expected_versions=["openroad=2025.2"],
    )

    assert [finding.level for finding in findings] == ["error", "error", "error"]
    messages = [finding.message for finding in findings]
    assert "required tool 'vivado' is not available" in messages
    assert "required tool 'yosys' is not in the inventory" in messages
    assert any(
        "tool 'openroad' version does not contain '2025.2'" in message for message in messages
    )


def test_main_writes_report_and_fails_for_missing_required_tool(tmp_path: Path) -> None:
    tool = _load_tool()

    def missing_runner(command: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError(command[0])

    out_path = tmp_path / "eda.json"

    exit_code = tool.main(["--out", str(out_path), "--require", "vivado"], runner=missing_runner)

    assert exit_code == 2
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["passed"] is False
    assert payload["findings"] == [
        {"level": "error", "message": "required tool 'vivado' is not available"}
    ]
