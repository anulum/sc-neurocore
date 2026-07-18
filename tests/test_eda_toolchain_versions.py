# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — EDA toolchain version inventory tests

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


def _load_tool() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "tools" / "eda_toolchain_versions.py"
    spec = importlib.util.spec_from_file_location("eda_toolchain_versions", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _runner(fixtures: dict[tuple[str, ...], subprocess.CompletedProcess[str]]):
    def run(command: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        key = tuple(command)
        if key not in fixtures:
            raise FileNotFoundError(command[0])
        return fixtures[key]

    return run


def _completed(stdout: str = "", stderr: str = "", returncode: int = 0):
    return subprocess.CompletedProcess(["tool"], returncode, stdout, stderr)


def test_collect_inventory_records_available_tools_and_environment() -> None:
    tool = _load_tool()

    report = tool.collect_eda_toolchain_versions(
        runner=_runner(
            {
                ("vivado", "-version"): _completed("Vivado v2025.2\n"),
                ("yosys", "--version"): _completed("Yosys 0.63+173\n"),
                ("openroad", "-version"): _completed("OpenROAD 2.0-test\n"),
                ("icepack", "--help"): _completed(
                    stderr="Usage: icepack [options] [input-file [output-file]]\n",
                    returncode=1,
                ),
            }
        ),
        version_lookup=lambda name: "3.0.1" if name == "pynq" else "",
        environ={
            "OPENROAD_IMAGE_DIGEST": "sha256:" + "a" * 64,
            "PDK": "sky130A",
            "PDK_ROOT": "/opt/pdk",
        },
    )

    assert report["schema_version"] == tool.SCHEMA_VERSION
    assert report["tools"]["vivado"]["available"] is True
    assert report["tools"]["vivado"]["version"] == "Vivado v2025.2"
    assert report["tools"]["yosys"]["version"] == "Yosys 0.63+173"
    assert report["tools"]["nextpnr_ice40"]["available"] is False
    assert report["tools"]["icepack"]["available"] is True
    assert report["tools"]["icepack"]["version"].startswith("Usage: icepack")
    assert report["tools"]["pynq"]["version"] == "3.0.1"
    assert report["environment"] == {
        "openroad_image_digest": "sha256:" + "a" * 64,
        "pdk": "sky130A",
        "pdk_root_set": True,
    }


def test_check_expectations_reports_missing_and_version_mismatch() -> None:
    tool = _load_tool()
    report = {
        "tools": {
            "vivado": {"available": True, "version": "Vivado v2025.2"},
            "openroad": {"available": False, "version": None},
        }
    }

    findings = tool.check_expectations(
        report,
        required=["vivado", "openroad", "yosys"],
        expected_versions=["vivado=v2024.2"],
    )

    assert [finding.level for finding in findings] == ["error", "error", "error"]
    assert "openroad" in findings[0].message
    assert "yosys" in findings[1].message
    assert "v2024.2" in findings[2].message


def test_main_writes_report_and_returns_failure_for_unmet_requirement(tmp_path: Path) -> None:
    tool = _load_tool()
    output = tmp_path / "eda.json"

    rc = tool.main(
        ["--out", str(output), "--require", "vivado", "--expect", "yosys=0.63"],
        runner=_runner({("yosys", "--version"): _completed("Yosys 0.63\n")}),
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert rc == 2
    assert payload["passed"] is False
    assert payload["tools"]["yosys"]["available"] is True
    assert payload["findings"][0]["message"] == "required tool 'vivado' is not available"


def test_main_accepts_matching_required_version(tmp_path: Path) -> None:
    tool = _load_tool()
    output = tmp_path / "eda.json"

    rc = tool.main(
        ["--out", str(output), "--require", "vivado", "--expect", "vivado=v2025.2"],
        runner=_runner({("vivado", "-version"): _completed("Vivado v2025.2\n")}),
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert rc == 0
    assert payload["passed"] is True
    assert payload["findings"] == []
