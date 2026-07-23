# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestQ1616Precision from former test_cosim_terman_wang.py

"""Focused suite: TestQ1616Precision from former test_cosim_terman_wang.py."""

from __future__ import annotations

from tests.cosim_terman_wang_support import *  # noqa: F403

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
