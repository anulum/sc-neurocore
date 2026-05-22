# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SC design optimisation tool tests

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest


def _tool() -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "tools" / "optimise_sc_design.py"
    spec = importlib.util.spec_from_file_location("optimise_sc_design", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_network(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "layers": [
                    {"id": "encoder", "mac_count": 256, "is_critical_path": True},
                    {"id": "decoder", "mac_count": 192},
                ]
            }
        ),
        encoding="utf-8",
    )


def _write_evidence(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "observations": [
                    {
                        "mac_count": 256,
                        "bitstream_length": 128,
                        "decorrelator": "LFSR",
                        "mode": "SC",
                        "precision_bits": 8,
                        "lfsr_polynomial": "x16+x15+x13+x4+1",
                        "luts_used": 260,
                        "power_mw": 1.1,
                        "latency_cycles": 128,
                        "accuracy_score": 0.999,
                        "is_critical_path": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def test_load_network_rejects_invalid_manifest(tmp_path: Path) -> None:
    tool = _tool()
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"layers": []}), encoding="utf-8")

    with pytest.raises(ValueError, match="non-empty layers list"):
        tool.load_network(bad)


def test_tool_writes_budgeted_json_plan(tmp_path: Path) -> None:
    tool = _tool()
    network = tmp_path / "network.json"
    evidence = tmp_path / "evidence.json"
    output = tmp_path / "plan.json"
    _write_network(network)
    _write_evidence(evidence)

    rc = tool.main(
        [
            "--network",
            str(network),
            "--evidence",
            str(evidence),
            "--out",
            str(output),
            "--target-name",
            "unit-fpga",
            "--max-luts",
            "10000",
            "--max-power-mw",
            "100",
            "--max-latency-cycles",
            "256",
        ]
    )

    plan = json.loads(output.read_text(encoding="utf-8"))
    assert rc == 0
    assert plan["target_name"] == "unit-fpga"
    assert plan["feasible"] is True
    assert {layer["id"] for layer in plan["layers"]} == {"encoder", "decoder"}
    assert plan["training_points"] > 0


def test_tool_exits_with_error_for_invalid_network(tmp_path: Path) -> None:
    tool = _tool()
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"layers": [{"id": "", "mac_count": 1}]}), encoding="utf-8")

    with pytest.raises(SystemExit) as exc:
        tool.main(
            [
                "--network",
                str(bad),
                "--max-luts",
                "10000",
                "--max-power-mw",
                "100",
            ]
        )

    assert exc.value.code == 2


def test_load_network_rejects_missing_required_layer_fields(tmp_path: Path) -> None:
    tool = _tool()
    missing_id = tmp_path / "missing_id.json"
    missing_mac = tmp_path / "missing_mac.json"
    missing_id.write_text(json.dumps({"layers": [{"mac_count": 1}]}), encoding="utf-8")
    missing_mac.write_text(json.dumps({"layers": [{"id": "encoder"}]}), encoding="utf-8")

    with pytest.raises(ValueError, match="missing id"):
        tool.load_network(missing_id)
    with pytest.raises(ValueError, match="missing mac_count"):
        tool.load_network(missing_mac)


def test_tool_writes_plan_to_stdout_when_out_not_provided(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    tool = _tool()
    network = tmp_path / "network.json"
    evidence = tmp_path / "evidence.json"
    _write_network(network)
    _write_evidence(evidence)

    rc = tool.main(
        [
            "--network",
            str(network),
            "--evidence",
            str(evidence),
            "--max-luts",
            "10000",
            "--max-power-mw",
            "100",
        ]
    )

    out = capsys.readouterr().out
    assert rc == 0
    assert '"layers"' in out
    assert '"target_name"' in out
