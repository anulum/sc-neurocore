# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header

"""Vivado-gated DCLS ZU3EG synthesis contract."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
HDL_FILES = [
    REPO_ROOT / "hdl" / "sc_dcls_axonal_delay.v",
    REPO_ROOT / "hdl" / "sc_dcls_tent_kernel.v",
    REPO_ROOT / "hdl" / "sc_dcls_layer_core.v",
]


def test_dcls_layer_core_vivado_ci_elaborates_for_zu3eg(tmp_path: Path) -> None:
    if os.environ.get("MIF_VIVADO_CI") != "1":
        pytest.skip("NEU-C.6 Vivado-CI gated: set MIF_VIVADO_CI=1 on a Vivado 2024.2 runner")
    vivado = shutil.which("vivado")
    if vivado is None:
        pytest.skip("NEU-C.6 Vivado-CI gated: Vivado executable is not installed on this host")

    tcl = tmp_path / "elaborate_dcls_zu3eg.tcl"
    tcl.write_text(
        "\n".join(
            [
                "create_project dcls_zu3eg dcls_zu3eg -part xczu3eg-sbva484-1-e -force",
                *(f"read_verilog -sv {path}" for path in HDL_FILES),
                "synth_design -top sc_dcls_layer_core -part xczu3eg-sbva484-1-e -mode out_of_context",
                "report_utilization -file dcls_utilization.rpt",
                "report_timing_summary -file dcls_timing.rpt",
                "quit",
                "",
            ]
        ),
        encoding="utf-8",
    )
    subprocess.run([vivado, "-mode", "batch", "-source", str(tcl)], cwd=tmp_path, check=True)
