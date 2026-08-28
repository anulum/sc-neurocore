# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Wilson-HR Q16.16 co-simulation and synthesis

"""Verify Wilson-HR fixed-point parity and committed RTL synthesis."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from tests.cosim_reference_wilson_hr import _wilson_hr_hand_spike_count
from tests.cosim_rtl_spike_execution import _verilog_spike_count_q1616
from tests.cosim_runtime import HAS_IVERILOG, _python_spike_count

ROOT = Path(__file__).resolve().parents[1]
RTL = ROOT / "hdl/formal/catalogue/sc_wilson_hr.v"
HAS_YOSYS = shutil.which("yosys") is not None


@pytest.mark.skipif(not HAS_IVERILOG, reason="Icarus Verilog not available")
class TestQ1616Precision:
    """Q16.16 Wilson-HR co-simulation fidelity."""

    @pytest.mark.parametrize(
        ("current", "expected_spikes"),
        ((0.0, 0), (0.08, 44), (0.1, 46), (0.14, 49), (0.2, 52)),
        ids=("silent", "low-cycle", "regular-cycle", "faster-cycle", "high-cycle"),
    )
    def test_wilson_hr_q1616_parity(self, current: float, expected_spikes: int) -> None:
        """Wilson-HR has exact three-way Q16.16 spike-count parity.

        The schema mirrors the maintained two-state polynomial cortical model:
        source capacitance ``C=0.8``, simultaneous four-stage RK4 over ``v`` and
        ``r``, continuous state, and sampled upward-crossing observation at ``v=0``.
        Over 5,000 steps all three paths must reproduce the enrolled operating points.
        """
        n_steps = 5000
        hand_spikes = _wilson_hr_hand_spike_count(n_steps, current)
        py_spikes = _python_spike_count("wilson_hr", n_steps, current)
        vlog_spikes = _verilog_spike_count_q1616("wilson_hr", n_steps, current)
        assert hand_spikes == py_spikes == vlog_spikes == expected_spikes, (
            f"Wilson-HR three-way mismatch at I={current}: hand={hand_spikes}, "
            f"schema={py_spikes}, verilog={vlog_spikes}"
        )


@pytest.mark.skipif(not HAS_YOSYS, reason="Yosys not available")
def test_yosys_synthesises_committed_rtl() -> None:
    """Require coarse synthesis of the committed source-model RTL."""
    result = subprocess.run(
        [
            "yosys",
            "-q",
            "-p",
            f"read_verilog {RTL}; synth -top sc_wilson_hr -run begin:coarse; check; stat",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
