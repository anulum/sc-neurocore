# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Synthesis evidence CLI tests

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest

from sc_neurocore.cli import main
from sc_neurocore.optimizer import load_observations


def _write_design(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "mac_count": 128,
                "bitstream_length": 64,
                "decorrelator": "LFSR",
                "mode": "SC",
                "precision_bits": 8,
                "lfsr_polynomial": "x16+x15+x13+x4+1",
            }
        ),
        encoding="utf-8",
    )


def _run_main(*argv: str) -> int:
    with mock.patch("sys.argv", ["sc-neurocore", *argv]):
        return main()


def test_collect_synthesis_command_writes_optimizer_evidence(tmp_path: Path) -> None:
    design = tmp_path / "design.json"
    utilisation = tmp_path / "utilisation.rpt"
    power = tmp_path / "power.rpt"
    output = tmp_path / "evidence.json"
    _write_design(design)
    utilisation.write_text("CLB LUTs | 512\nLatency: 64 cycles\n", encoding="utf-8")
    power.write_text("Total On-Chip Power (mW): 25.0\n", encoding="utf-8")

    rc = _run_main(
        "collect-synthesis",
        "--design",
        str(design),
        "--utilisation",
        str(utilisation),
        "--power",
        str(power),
        "--accuracy-score",
        "0.99",
        "--out",
        str(output),
    )

    observations = load_observations(output)
    assert rc == 0
    assert observations[0].luts_used == 512
    assert observations[0].power_mw == 25.0


def test_collect_synthesis_command_reports_missing_required_args(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = _run_main("collect-synthesis")

    assert rc == 1
    assert "--design" in capsys.readouterr().out


def test_collect_synthesis_rejects_non_object_design_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    design = tmp_path / "design.json"
    utilisation = tmp_path / "utilisation.rpt"
    power = tmp_path / "power.rpt"
    design.write_text("[]", encoding="utf-8")
    utilisation.write_text("CLB LUTs | 512\nLatency: 64 cycles\n", encoding="utf-8")
    power.write_text("Total On-Chip Power (mW): 25.0\n", encoding="utf-8")

    rc = _run_main(
        "collect-synthesis",
        "--design",
        str(design),
        "--utilisation",
        str(utilisation),
        "--power",
        str(power),
        "--accuracy-score",
        "0.99",
    )

    assert rc == 1
    assert "must be a JSON object" in capsys.readouterr().out


def test_collect_synthesis_accepts_utilization_alias(tmp_path: Path) -> None:
    design = tmp_path / "design.json"
    utilisation = tmp_path / "utilisation.rpt"
    power = tmp_path / "power.rpt"
    output = tmp_path / "evidence.json"
    _write_design(design)
    utilisation.write_text("CLB LUTs | 256\nLatency: 32 cycles\n", encoding="utf-8")
    power.write_text("Total On-Chip Power (mW): 12.5\n", encoding="utf-8")

    rc = _run_main(
        "collect-synthesis",
        "--design",
        str(design),
        "--utilization",
        str(utilisation),
        "--power",
        str(power),
        "--accuracy-score",
        "0.97",
        "--out",
        str(output),
    )

    observations = load_observations(output)
    assert rc == 0
    assert observations[0].luts_used == 256


def test_collect_synthesis_rejects_partial_energy_metadata(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    design = tmp_path / "design.json"
    utilisation = tmp_path / "utilisation.rpt"
    power = tmp_path / "power.rpt"
    _write_design(design)
    utilisation.write_text("CLB LUTs | 256\nLatency: 32 cycles\n", encoding="utf-8")
    power.write_text("Total On-Chip Power (mW): 12.5\n", encoding="utf-8")

    rc = _run_main(
        "collect-synthesis",
        "--design",
        str(design),
        "--utilisation",
        str(utilisation),
        "--power",
        str(power),
        "--accuracy-score",
        "0.97",
        "--clock-mhz",
        "100.0",
    )

    assert rc == 1
    assert "energy requires both --clock-mhz and --inferences-per-run" in capsys.readouterr().out


def test_collect_synthesis_rejects_non_positive_energy_parameters(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    design = tmp_path / "design.json"
    utilisation = tmp_path / "utilisation.rpt"
    power = tmp_path / "power.rpt"
    _write_design(design)
    utilisation.write_text("CLB LUTs | 256\nLatency: 32 cycles\n", encoding="utf-8")
    power.write_text("Total On-Chip Power (mW): 12.5\n", encoding="utf-8")

    rc_clock = _run_main(
        "collect-synthesis",
        "--design",
        str(design),
        "--utilisation",
        str(utilisation),
        "--power",
        str(power),
        "--accuracy-score",
        "0.97",
        "--clock-mhz",
        "0",
        "--inferences-per-run",
        "1",
    )
    rc_inf = _run_main(
        "collect-synthesis",
        "--design",
        str(design),
        "--utilisation",
        str(utilisation),
        "--power",
        str(power),
        "--accuracy-score",
        "0.97",
        "--clock-mhz",
        "100.0",
        "--inferences-per-run",
        "0",
    )

    output = capsys.readouterr().out
    assert rc_clock == 1
    assert rc_inf == 1
    assert "clock_mhz must be positive" in output
    assert "inferences_per_run must be positive" in output
