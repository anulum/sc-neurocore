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
    assert "--antagonistic-pair" in without_sby
    assert "0,1" in without_sby
    assert "--temporal-separation" in without_sby
    assert "0,1,2" in without_sby
    assert "--coactivation-cap" in without_sby
    assert "1" in without_sby
    assert "--population-silence" in without_sby
    assert "2,2" in without_sby
    assert "--population-inactivity" in without_sby
    assert "3" in without_sby
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
    assert payload["coverage_manifest"]["output_width"] == 2
    assert payload["coverage_manifest"]["covered_outputs"] == [0, 1]
    assert len(payload["coverage_manifest"]["reports"]) == 2
    for output_index in range(2):
        report_path = output / f"output_{output_index}" / "formal_rate_bound_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report["rate_bound"]["output_index"] == output_index
        assert report["refractory"]["output_index"] == output_index
        assert report["antagonistic_exclusion"]["output_a"] == 0
        assert report["antagonistic_exclusion"]["output_b"] == 1
        assert report["temporal_separation"]["output_a"] == 0
        assert report["temporal_separation"]["output_b"] == 1
        assert report["temporal_separation"]["separation_cycles"] == 2
        assert report["population_coactivation"]["max_active_outputs"] == 1
        assert report["population_silence"]["trigger_active_outputs"] == 2
        assert report["population_silence"]["silence_cycles"] == 2
        assert report["population_inactivity"]["max_silent_cycles"] == 3
        assert report["symbiyosys"]["requested"] is False
        assert report["symbiyosys"]["status"] == "not_requested"


def test_main_keeps_population_inactivity_for_single_output_fixture(
    tmp_path: Path, monkeypatch: Any
) -> None:
    tool = _load_tool()
    output = tmp_path / "formal"
    summary = tmp_path / "summary.json"
    monkeypatch.setattr(tool.shutil, "which", lambda name: None if name == "sby" else None)

    exit_code = tool.main(
        ["--output", str(output), "--summary", str(summary), "--output-width", "1"]
    )

    assert exit_code == 0
    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["coverage_manifest"]["required_outputs"] == [0]
    report = json.loads(
        (output / "output_0" / "formal_rate_bound_report.json").read_text(encoding="utf-8")
    )
    assert report["network"]["output_width"] == 1
    assert report["antagonistic_exclusion"] is None
    assert report["temporal_separation"] is None
    assert report["population_coactivation"] is None
    assert report["population_silence"] is None
    assert report["population_inactivity"]["max_silent_cycles"] == 3
    assert report["artifacts"]["antagonistic_sva"] is None
    assert report["artifacts"]["temporal_sva"] is None
    assert report["artifacts"]["population_sva"] is None
    assert report["artifacts"]["population_silence_sva"] is None
    assert report["artifacts"]["population_inactivity_sva"].endswith(
        "dense_lif_frontier_fixture_population_inactivity.sv"
    )


def test_main_fails_when_any_output_report_is_missing(tmp_path: Path, monkeypatch: Any) -> None:
    tool = _load_tool()
    output = tmp_path / "formal"
    summary = tmp_path / "summary.json"

    def fake_runner(command: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        output_index = int(command[command.index("--output-index") + 1])
        if output_index == 1:
            return subprocess.CompletedProcess(command, 0, stdout="missing report\n", stderr="")
        report_path = Path(command[command.index("--out") + 1])
        artifact_root = Path(command[command.index("--output") + 1])
        artifact_root.mkdir(parents=True, exist_ok=True)
        paths = {
            "rtl": artifact_root / "dense_lif_frontier_fixture.v",
            "sva": artifact_root / "dense_lif_frontier_fixture_rate_bound.sv",
            "rate_sva": artifact_root / "dense_lif_frontier_fixture_rate_bound.sv",
            "refractory_sva": artifact_root / "dense_lif_frontier_fixture_refractory.sv",
            "antagonistic_sva": artifact_root / "dense_lif_frontier_fixture_antagonistic.sv",
            "temporal_sva": artifact_root / "dense_lif_frontier_fixture_temporal_separation.sv",
            "population_sva": artifact_root
            / "dense_lif_frontier_fixture_population_coactivation.sv",
            "population_silence_sva": artifact_root
            / "dense_lif_frontier_fixture_population_silence.sv",
            "population_inactivity_sva": artifact_root
            / "dense_lif_frontier_fixture_population_inactivity.sv",
            "formal_bundle": artifact_root / "dense_lif_frontier_fixture_formal_bundle.sv",
            "sby": artifact_root / "dense_lif_frontier_fixture.sby",
            "report": report_path,
        }
        for path in paths.values():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("artifact\n", encoding="utf-8")
        report_path.write_text(
            json.dumps(_valid_report_payload(output_index, paths)) + "\n",
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
    assert "output 1" in payload["error"]
    assert "formal_rate_bound_report.json" in payload["error"]


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
                    "antagonistic_exclusion": None,
                    "temporal_separation": None,
                    "population_coactivation": None,
                    "population_silence": None,
                    "artifacts": {
                        "rtl": str(tmp_path / "outside.v"),
                        "sva": str(tmp_path / "outside.sv"),
                        "rate_sva": str(tmp_path / "outside.sv"),
                        "refractory_sva": None,
                        "antagonistic_sva": None,
                        "temporal_sva": None,
                        "population_sva": None,
                        "population_silence_sva": None,
                        "population_inactivity_sva": None,
                        "formal_bundle": str(tmp_path / "outside_bundle.sv"),
                        "sby": str(tmp_path / "outside.sby"),
                        "report": str(report_path),
                    },
                    "replay": None,
                    "rate_replay": None,
                    "refractory_replay": None,
                    "antagonistic_replay": None,
                    "temporal_replay": None,
                    "population_replay": None,
                    "population_silence_replay": None,
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


def test_main_fails_when_report_antagonistic_metadata_exceeds_manifest_width(
    tmp_path: Path, monkeypatch: Any
) -> None:
    tool = _load_tool()
    output = tmp_path / "formal"
    summary = tmp_path / "summary.json"

    def fake_runner(command: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        output_index = int(command[command.index("--output-index") + 1])
        report_path = Path(command[command.index("--out") + 1])
        artifact_root = Path(command[command.index("--output") + 1])
        artifact_root.mkdir(parents=True, exist_ok=True)
        paths = {
            "rtl": artifact_root / "dense_lif_frontier_fixture.v",
            "sva": artifact_root / "dense_lif_frontier_fixture_rate_bound.sv",
            "rate_sva": artifact_root / "dense_lif_frontier_fixture_rate_bound.sv",
            "refractory_sva": artifact_root / "dense_lif_frontier_fixture_refractory.sv",
            "antagonistic_sva": artifact_root / "dense_lif_frontier_fixture_antagonistic.sv",
            "temporal_sva": artifact_root / "dense_lif_frontier_fixture_temporal_separation.sv",
            "population_sva": artifact_root
            / "dense_lif_frontier_fixture_population_coactivation.sv",
            "population_silence_sva": artifact_root
            / "dense_lif_frontier_fixture_population_silence.sv",
            "population_inactivity_sva": artifact_root
            / "dense_lif_frontier_fixture_population_inactivity.sv",
            "formal_bundle": artifact_root / "dense_lif_frontier_fixture_formal_bundle.sv",
            "sby": artifact_root / "dense_lif_frontier_fixture.sby",
            "report": report_path,
        }
        for path in paths.values():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("artifact\n", encoding="utf-8")
        report = _valid_report_payload(output_index, paths)
        network = report["network"]
        assert isinstance(network, dict)
        network["output_width"] = 3
        antagonistic = report["antagonistic_exclusion"]
        assert isinstance(antagonistic, dict)
        antagonistic["output_b"] = 2
        report_path.write_text(json.dumps(report) + "\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="ok\n", stderr="")

    monkeypatch.setattr(tool.shutil, "which", lambda name: None if name == "sby" else None)

    exit_code = tool.main(
        [
            "--output",
            str(output),
            "--summary",
            str(summary),
        ],
        runner=fake_runner,
    )

    assert exit_code == 1
    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert payload["passed"] is False
    assert "network.output_width does not match manifest output_width" in payload["error"]


def test_main_fails_when_report_temporal_metadata_differs_from_manifest(
    tmp_path: Path, monkeypatch: Any
) -> None:
    tool = _load_tool()
    output = tmp_path / "formal"
    summary = tmp_path / "summary.json"

    def fake_runner(command: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        output_index = int(command[command.index("--output-index") + 1])
        report_path = Path(command[command.index("--out") + 1])
        artifact_root = Path(command[command.index("--output") + 1])
        artifact_root.mkdir(parents=True, exist_ok=True)
        paths = {
            "rtl": artifact_root / "dense_lif_frontier_fixture.v",
            "sva": artifact_root / "dense_lif_frontier_fixture_rate_bound.sv",
            "rate_sva": artifact_root / "dense_lif_frontier_fixture_rate_bound.sv",
            "refractory_sva": artifact_root / "dense_lif_frontier_fixture_refractory.sv",
            "antagonistic_sva": artifact_root / "dense_lif_frontier_fixture_antagonistic.sv",
            "temporal_sva": artifact_root / "dense_lif_frontier_fixture_temporal_separation.sv",
            "population_sva": artifact_root
            / "dense_lif_frontier_fixture_population_coactivation.sv",
            "population_silence_sva": artifact_root
            / "dense_lif_frontier_fixture_population_silence.sv",
            "population_inactivity_sva": artifact_root
            / "dense_lif_frontier_fixture_population_inactivity.sv",
            "formal_bundle": artifact_root / "dense_lif_frontier_fixture_formal_bundle.sv",
            "sby": artifact_root / "dense_lif_frontier_fixture.sby",
            "report": report_path,
        }
        for path in paths.values():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("artifact\n", encoding="utf-8")
        report = _valid_report_payload(output_index, paths)
        temporal = report["temporal_separation"]
        assert isinstance(temporal, dict)
        temporal["separation_cycles"] = 3
        report_path.write_text(json.dumps(report) + "\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="ok\n", stderr="")

    monkeypatch.setattr(tool.shutil, "which", lambda name: None if name == "sby" else None)

    exit_code = tool.main(
        ["--output", str(output), "--summary", str(summary)],
        runner=fake_runner,
    )

    assert exit_code == 1
    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert payload["passed"] is False
    assert "temporal_separation does not match manifest temporal_separation" in payload["error"]


def test_main_fails_when_report_population_metadata_differs_from_manifest(
    tmp_path: Path, monkeypatch: Any
) -> None:
    tool = _load_tool()
    output = tmp_path / "formal"
    summary = tmp_path / "summary.json"

    def fake_runner(command: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        output_index = int(command[command.index("--output-index") + 1])
        report_path = Path(command[command.index("--out") + 1])
        artifact_root = Path(command[command.index("--output") + 1])
        artifact_root.mkdir(parents=True, exist_ok=True)
        paths = {
            "rtl": artifact_root / "dense_lif_frontier_fixture.v",
            "sva": artifact_root / "dense_lif_frontier_fixture_rate_bound.sv",
            "rate_sva": artifact_root / "dense_lif_frontier_fixture_rate_bound.sv",
            "refractory_sva": artifact_root / "dense_lif_frontier_fixture_refractory.sv",
            "antagonistic_sva": artifact_root / "dense_lif_frontier_fixture_antagonistic.sv",
            "temporal_sva": artifact_root / "dense_lif_frontier_fixture_temporal_separation.sv",
            "population_sva": artifact_root
            / "dense_lif_frontier_fixture_population_coactivation.sv",
            "population_silence_sva": artifact_root
            / "dense_lif_frontier_fixture_population_silence.sv",
            "population_inactivity_sva": artifact_root
            / "dense_lif_frontier_fixture_population_inactivity.sv",
            "formal_bundle": artifact_root / "dense_lif_frontier_fixture_formal_bundle.sv",
            "sby": artifact_root / "dense_lif_frontier_fixture.sby",
            "report": report_path,
        }
        for path in paths.values():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("artifact\n", encoding="utf-8")
        report = _valid_report_payload(output_index, paths)
        population = report["population_coactivation"]
        assert isinstance(population, dict)
        population["max_active_outputs"] = 2
        report_path.write_text(json.dumps(report) + "\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="ok\n", stderr="")

    monkeypatch.setattr(tool.shutil, "which", lambda name: None if name == "sby" else None)

    exit_code = tool.main(
        ["--output", str(output), "--summary", str(summary)],
        runner=fake_runner,
    )

    assert exit_code == 1
    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert payload["passed"] is False
    assert "population_coactivation does not match manifest coactivation_cap" in payload["error"]


def test_main_fails_when_report_population_silence_differs_from_manifest(
    tmp_path: Path, monkeypatch: Any
) -> None:
    tool = _load_tool()
    output = tmp_path / "formal"
    summary = tmp_path / "summary.json"

    def fake_runner(command: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        output_index = int(command[command.index("--output-index") + 1])
        report_path = Path(command[command.index("--out") + 1])
        artifact_root = Path(command[command.index("--output") + 1])
        artifact_root.mkdir(parents=True, exist_ok=True)
        paths = {
            "rtl": artifact_root / "dense_lif_frontier_fixture.v",
            "sva": artifact_root / "dense_lif_frontier_fixture_rate_bound.sv",
            "rate_sva": artifact_root / "dense_lif_frontier_fixture_rate_bound.sv",
            "refractory_sva": artifact_root / "dense_lif_frontier_fixture_refractory.sv",
            "antagonistic_sva": artifact_root / "dense_lif_frontier_fixture_antagonistic.sv",
            "temporal_sva": artifact_root / "dense_lif_frontier_fixture_temporal_separation.sv",
            "population_sva": artifact_root
            / "dense_lif_frontier_fixture_population_coactivation.sv",
            "population_silence_sva": artifact_root
            / "dense_lif_frontier_fixture_population_silence.sv",
            "population_inactivity_sva": artifact_root
            / "dense_lif_frontier_fixture_population_inactivity.sv",
            "formal_bundle": artifact_root / "dense_lif_frontier_fixture_formal_bundle.sv",
            "sby": artifact_root / "dense_lif_frontier_fixture.sby",
            "report": report_path,
        }
        for path in paths.values():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("artifact\n", encoding="utf-8")
        report = _valid_report_payload(output_index, paths)
        silence = report["population_silence"]
        assert isinstance(silence, dict)
        silence["silence_cycles"] = 3
        report_path.write_text(json.dumps(report) + "\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="ok\n", stderr="")

    monkeypatch.setattr(tool.shutil, "which", lambda name: None if name == "sby" else None)

    exit_code = tool.main(
        ["--output", str(output), "--summary", str(summary)],
        runner=fake_runner,
    )

    assert exit_code == 1
    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert payload["passed"] is False
    assert "population_silence does not match manifest population_silence" in payload["error"]


def _valid_report_payload(output_index: int, paths: dict[str, Path]) -> dict[str, object]:
    return {
        "schema_version": "sc-neurocore.formal-network-rate-bound.v0.1",
        "network": {
            "name": "dense_lif_frontier_fixture",
            "input_width": 3,
            "output_width": 2,
            "state_width": 16,
            "timestep_name": "sample_valid",
            "output_signal": "spike_out",
            "clock_name": "clk",
            "reset_name": "rst_n",
        },
        "rate_bound": {
            "name": f"output{output_index}_rate_bound",
            "output_index": output_index,
            "window_cycles": 8,
            "max_spikes": 4,
        },
        "refractory": {
            "name": f"output{output_index}_refractory",
            "output_index": output_index,
            "refractory_cycles": 2,
        },
        "antagonistic_exclusion": {
            "name": "output0_output1_exclusion",
            "output_a": 0,
            "output_b": 1,
        },
        "temporal_separation": {
            "name": "output0_output1_temporal_separation",
            "output_a": 0,
            "output_b": 1,
            "separation_cycles": 2,
        },
        "population_coactivation": {
            "name": "population_coactivation_cap",
            "max_active_outputs": 1,
        },
        "population_silence": {
            "name": "population_silence_after_coactivation",
            "trigger_active_outputs": 2,
            "silence_cycles": 2,
        },
        "population_inactivity": {
            "name": "population_inactivity_bound",
            "max_silent_cycles": 3,
        },
        "artifacts": {key: str(path) for key, path in paths.items()},
        "replay": None,
        "rate_replay": None,
        "refractory_replay": None,
        "antagonistic_replay": None,
        "temporal_replay": None,
        "population_replay": None,
        "population_silence_replay": None,
        "population_inactivity_replay": None,
        "symbiyosys": {
            "requested": False,
            "status": "not_requested",
            "command": None,
            "returncode": None,
            "stdout": "",
            "stderr": "",
            "sby": str(paths["sby"]),
        },
    }
