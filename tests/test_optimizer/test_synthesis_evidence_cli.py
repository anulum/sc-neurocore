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
