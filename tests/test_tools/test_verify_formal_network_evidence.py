# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — tests for tools/verify_formal_network_evidence.py

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


REPO = Path(__file__).resolve().parents[2]
TOOL = REPO / "tools/verify_formal_network_evidence.py"


def _load_tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("verify_formal_network_evidence", TOOL)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_command_requests_symbiyosys_only_when_available(tmp_path: Path) -> None:
    tool = _load_tool()
    config = tool.FormalEvidenceConfig(output=tmp_path / "formal")

    without_sby = tool.build_formal_verify_command(config, sby_available=False)
    with_sby = tool.build_formal_verify_command(config, sby_available=True)

    assert "--run-symbiyosys" not in without_sby
    assert "--run-symbiyosys" in with_sby
    assert without_sby[0:3] == [sys.executable, "-m", "sc_neurocore.cli"]
    assert "--out" in without_sby
    assert str(config.report_path) in without_sby


def test_main_generates_and_validates_report_with_artifact_root(
    tmp_path: Path, monkeypatch: Any
) -> None:
    tool = _load_tool()
    output = tmp_path / "formal"
    summary = tmp_path / "summary.json"
    monkeypatch.setattr(tool.shutil, "which", lambda name: None if name == "sby" else None)

    exit_code = tool.main(["--output", str(output), "--summary", str(summary)])

    assert exit_code == 0
    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["symbiyosys_requested"] is False
    assert payload["report"] == str(output / "formal_rate_bound_report.json")
    assert payload["artifact_root"] == str(output)
    report = json.loads((output / "formal_rate_bound_report.json").read_text(encoding="utf-8"))
    assert report["symbiyosys"]["requested"] is False
    assert report["symbiyosys"]["status"] == "not_requested"


def test_main_fails_when_generated_report_fails_artifact_root_validation(
    tmp_path: Path, monkeypatch: Any
) -> None:
    tool = _load_tool()
    output = tmp_path / "formal"
    summary = tmp_path / "summary.json"

    def fake_runner(command: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        report_path = Path(command[command.index("--out") + 1])
        report_path.parent.mkdir(parents=True, exist_ok=True)
        for path in (
            tmp_path / "outside.v",
            tmp_path / "outside.sv",
            tmp_path / "outside_bundle.sv",
            tmp_path / "outside.sby",
        ):
            path.write_text("outside artifact root\n", encoding="utf-8")
        report_path.write_text(
            json.dumps(
                {
                    "schema_version": "sc-neurocore.formal-network-rate-bound.v0.1",
                    "network": {
                        "name": "dense_lif_frontier_fixture",
                        "input_width": 3,
                        "output_width": 1,
                        "state_width": 16,
                        "timestep_name": "sample_valid",
                        "output_signal": "spike_out",
                    },
                    "rate_bound": {
                        "name": "output0_rate_bound",
                        "output_index": 0,
                        "window_cycles": 8,
                        "max_spikes": 4,
                    },
                    "refractory": None,
                    "artifacts": {
                        "rtl": str(tmp_path / "outside.v"),
                        "sva": str(tmp_path / "outside.sv"),
                        "rate_sva": str(tmp_path / "outside.sv"),
                        "refractory_sva": None,
                        "formal_bundle": str(tmp_path / "outside_bundle.sv"),
                        "sby": str(tmp_path / "outside.sby"),
                        "report": str(report_path),
                    },
                    "replay": None,
                    "rate_replay": None,
                    "refractory_replay": None,
                    "symbiyosys": {
                        "requested": False,
                        "status": "not_requested",
                        "command": None,
                        "returncode": None,
                        "stdout": "",
                        "stderr": "",
                        "sby": str(tmp_path / "outside.sby"),
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, stdout="ok\n", stderr="")

    monkeypatch.setattr(tool.shutil, "which", lambda name: None if name == "sby" else None)

    exit_code = tool.main(
        ["--output", str(output), "--summary", str(summary)],
        runner=fake_runner,
    )

    assert exit_code == 1
    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert payload["passed"] is False
    assert "outside artifact_root" in payload["error"]
