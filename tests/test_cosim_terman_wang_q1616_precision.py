# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Terman-Wang Q16.16 co-simulation and synthesis tests

"""Focused suite: TestQ1616Precision from former test_cosim_terman_wang.py."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tests.cosim_terman_wang_support import (
    HAS_IVERILOG,
    _python_spike_count,
    _terman_wang_hand_spike_count,
    _verilog_spike_count_q1616,
)
from tests.toolchain_support import require_executable

_ROOT = Path(__file__).resolve().parents[1]
_RTL = _ROOT / "hdl/formal/catalogue/sc_terman_wang.v"
_SBY = _ROOT / "hdl/formal/catalogue/sc_terman_wang.sby"


def test_yosys_synthesises_committed_rtl() -> None:
    completed = subprocess.run(
        [
            require_executable("yosys"),
            "-q",
            "-p",
            f"read_verilog {_RTL}; synth -top sc_terman_wang -run begin:coarse; check; stat",
        ],
        cwd=_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stderr


def test_symbiyosys_proves_the_committed_reset_contract(tmp_path: Path) -> None:
    """The public catalogue SBY entrypoint must execute and reach PASS."""
    engine = next(
        line for line in _SBY.read_text(encoding="utf-8").splitlines() if line.startswith("smtbmc ")
    )
    require_executable(engine.split(maxsplit=1)[1])
    output = tmp_path / "terman_wang_bmc"
    completed = subprocess.run(
        [require_executable("sby"), "-f", "-d", str(output), _SBY.name],
        cwd=_SBY.parent,
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    status = (output / "status").read_text(encoding="utf-8").split()
    assert status and status[0] == "PASS"


@pytest.mark.skipif(not HAS_IVERILOG, reason="Icarus Verilog not available")
class TestQ1616Precision:
    """Q16.16 Terman-Wang co-simulation fidelity."""

    @pytest.mark.parametrize(
        ("current", "expected_spikes"),
        ((-1.0, 0), (0.0, 1), (0.5, 3)),
        ids=("silent", "single-crossing", "oscillatory-train"),
    )
    def test_terman_wang_q1616_parity(self, current: float, expected_spikes: int) -> None:
        """Terman-Wang has exact three-way Q16.16 spike-count parity.

        The enrolled schema mirrors the maintained two-state LEGION oscillator:
        simultaneous four-stage RK4 over the cubic fast nullcline and ``tanh``-gated
        slow recovery, rising-edge ``v >= v_peak`` detection, and no reset. The
        transcendental gate makes raw state bit identity non-portable, so the declared
        observable is the robust silent/single/train crossing count: 0, 1, and 3 at
        ``I=-1.0``, ``0.0``, and ``0.5`` respectively over 8,000 steps.
        """
        n_steps = 8000
        hand_spikes = _terman_wang_hand_spike_count(n_steps, current)
        py_spikes = _python_spike_count("terman_wang", n_steps, current)
        vlog_spikes = _verilog_spike_count_q1616("terman_wang", n_steps, current)
        assert hand_spikes == py_spikes == vlog_spikes == expected_spikes, (
            f"Terman-Wang three-way mismatch at I={current}: hand={hand_spikes}, "
            f"schema={py_spikes}, verilog={vlog_spikes}"
        )
