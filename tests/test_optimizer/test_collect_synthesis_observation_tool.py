# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Synthesis observation collection tool tests

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from sc_neurocore.optimizer import load_observations


def _tool() -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "tools" / "collect_synthesis_observation.py"
    spec = importlib.util.spec_from_file_location("collect_synthesis_observation", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_design(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "mac_count": 256,
                "bitstream_length": 128,
                "decorrelator": "LFSR",
                "mode": "SC",
                "precision_bits": 8,
                "lfsr_polynomial": "x16+x15+x13+x4+1",
                "is_critical_path": True,
            }
        ),
        encoding="utf-8",
    )


def test_tool_writes_optimizer_observation_and_energy_payload(tmp_path: Path) -> None:
    tool = _tool()
    design = tmp_path / "design.json"
    utilisation = tmp_path / "utilisation.rpt"
    power = tmp_path / "power.rpt"
    output = tmp_path / "evidence.json"
    _write_design(design)
    utilisation.write_text("CLB LUTs | 1,024\nLatency: 128 cycles\n", encoding="utf-8")
    power.write_text("Total On-Chip Power (W): 0.125\n", encoding="utf-8")

    rc = tool.main(
        [
            "--design",
            str(design),
            "--utilisation",
            str(utilisation),
            "--power",
            str(power),
            "--accuracy-score",
            "0.991",
            "--clock-mhz",
            "100",
            "--inferences-per-run",
            "2",
            "--out",
            str(output),
        ]
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    observations = load_observations(output)
    assert rc == 0
    assert observations[0].luts_used == 1024
    assert observations[0].power_mw == 125.0
    assert observations[0].latency_cycles == 128
    assert payload["energy"]["energy_uj_per_inference"] == pytest.approx(0.08)


def test_tool_requires_complete_energy_metadata(tmp_path: Path) -> None:
    tool = _tool()
    design = tmp_path / "design.json"
    utilisation = tmp_path / "utilisation.rpt"
    power = tmp_path / "power.rpt"
    _write_design(design)
    utilisation.write_text("CLB LUTs | 42\nLatency: 16 cycles\n", encoding="utf-8")
    power.write_text("Total On-Chip Power (mW): 2.5\n", encoding="utf-8")

    with pytest.raises(SystemExit) as exc:
        tool.main(
            [
                "--design",
                str(design),
                "--utilisation",
                str(utilisation),
                "--power",
                str(power),
                "--accuracy-score",
                "0.99",
                "--clock-mhz",
                "100",
            ]
        )

    assert exc.value.code == 2
