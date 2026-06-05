# SPDX-License-Identifier: AGPL-3.0-or-later
"""Vivado-gated synthesis contract for the ADC-to-spike quantiser."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
HDL = REPO_ROOT / "hdl" / "sensors" / "adc_to_spike_quantiser.v"


def test_adc_to_spike_quantiser_vivado_ci_elaborates() -> None:
    if os.environ.get("MIF_VIVADO_CI") != "1":
        pytest.skip("NEU-C.5 Vivado-CI gated: set MIF_VIVADO_CI=1 on a Vivado 2024.2 runner")
    vivado = shutil.which("vivado")
    if vivado is None:
        pytest.skip("NEU-C.5 Vivado-CI gated: Vivado executable is not installed on this host")

    subprocess.run(
        [
            vivado,
            "-mode",
            "batch",
            "-source",
            "tools/vivado_elaborate_adc_to_spike.tcl",
            "-tclargs",
            str(HDL),
        ],
        cwd=REPO_ROOT,
        check=True,
    )
