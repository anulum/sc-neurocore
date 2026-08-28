# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — retained resetting Wilson-HR Q16.16 co-simulation

"""Hand/schema/RTL event-count parity for the retained SC recurrence."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from tests.cosim_reference_wilson_hr import _sc_resetting_wilson_hr_rk4_features
from tests.cosim_rtl_spike_execution import _verilog_spike_count_q1616
from tests.cosim_runtime import HAS_IVERILOG, _python_spike_count

ROOT = Path(__file__).resolve().parents[1]
RTL = ROOT / "hdl/formal/catalogue/sc_resetting_wilson_hr.v"
HAS_YOSYS = shutil.which("yosys") is not None


@pytest.mark.skipif(not HAS_IVERILOG, reason="Icarus Verilog not available")
@pytest.mark.parametrize(
    ("current", "expected_events"),
    ((0.0, 0), (2.0, 1), (10.0, 4)),
    ids=("silent", "single-event", "repetitive-events"),
)
def test_sc_resetting_wilson_hr_q1616_parity(current: float, expected_events: int) -> None:
    """Require exact event counts from hand, schema, and generated Q16.16 RTL."""
    n_steps = 5_000
    hand_features = _sc_resetting_wilson_hr_rk4_features(current=current, dt=0.05, steps=n_steps)
    hand_events = int(hand_features["spike_count"])
    schema_events = _python_spike_count("sc_resetting_wilson_hr", n_steps, current)
    rtl_events = _verilog_spike_count_q1616("sc_resetting_wilson_hr", n_steps, current)
    assert hand_events == schema_events == rtl_events == expected_events


@pytest.mark.skipif(not HAS_YOSYS, reason="Yosys not available")
def test_yosys_synthesises_committed_rtl() -> None:
    """Require coarse synthesis of the committed retained-SC RTL."""
    result = subprocess.run(
        [
            "yosys",
            "-q",
            "-p",
            (
                f"read_verilog {RTL}; synth -top sc_resetting_wilson_hr "
                "-run begin:coarse; check; stat"
            ),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
