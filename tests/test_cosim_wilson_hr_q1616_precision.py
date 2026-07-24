# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestQ1616Precision from former test_cosim_wilson_hr.py

"""Focused suite: TestQ1616Precision from former test_cosim_wilson_hr.py."""

from __future__ import annotations

from tests.cosim_wilson_hr_support import *  # noqa: F403


@pytest.mark.skipif(not HAS_IVERILOG, reason="Icarus Verilog not available")
class TestQ1616Precision:
    """Q16.16 Wilson-HR co-simulation fidelity."""

    @pytest.mark.parametrize(
        ("current", "expected_spikes"),
        ((0.0, 0), (2.0, 1), (10.0, 4)),
        ids=("silent", "single-spike", "four-spike-train"),
    )
    def test_wilson_hr_q1616_parity(self, current: float, expected_spikes: int) -> None:
        """Wilson-HR has exact three-way Q16.16 spike-count parity.

        The schema mirrors the maintained two-state polynomial cortical model:
        simultaneous four-stage RK4 over ``v`` and ``r``, level detection at
        ``v >= v_peak``, and a hard ``v = -0.7`` reset that preserves the candidate
        recovery state. Over 5,000 steps the hand model, schema runner, and emitted
        RTL reproduce the silent, single-spike, and four-spike operating points.
        """
        n_steps = 5000
        hand_spikes = _wilson_hr_hand_spike_count(n_steps, current)
        py_spikes = _python_spike_count("wilson_hr", n_steps, current)
        vlog_spikes = _verilog_spike_count_q1616("wilson_hr", n_steps, current)
        assert hand_spikes == py_spikes == vlog_spikes == expected_spikes, (
            f"Wilson-HR three-way mismatch at I={current}: hand={hand_spikes}, "
            f"schema={py_spikes}, verilog={vlog_spikes}"
        )
