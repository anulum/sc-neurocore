# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header

from __future__ import annotations
from pathlib import Path
import subprocess
import pytest

from tests.toolchain_support import require_executable

ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize("stem", ["benda_herz", "sc_stochastic_rate_adaptation"])
def test_yosys_synthesizes_catalogue_unit(stem: str) -> None:
    rtl = ROOT / f"hdl/formal/catalogue/{stem}.v"
    result = subprocess.run(
        [require_executable("yosys"), "-q", "-p", f"read_verilog {rtl}; synth -top {stem}; check"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
