# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestQ1616Precision from former test_cosim_fitzhugh_rinzel.py

"""Focused suite: TestQ1616Precision from former test_cosim_fitzhugh_rinzel.py."""

from __future__ import annotations

from tests.cosim_fitzhugh_rinzel_support import *  # noqa: F403

@pytest.mark.skipif(not HAS_IVERILOG, reason="Icarus Verilog not available")
class TestQ1616Precision:
    """Q16.16 FitzHugh-Rinzel co-simulation fidelity."""

    @pytest.mark.parametrize(
        ("current", "expected_spikes"),
        ((0.4, 7), (0.5, 8), (0.6, 8)),
        ids=("I=0.4", "I=0.5", "I=0.6"),
    )
    def test_fitzhugh_rinzel_q1616_parity(self, current: float, expected_spikes: int) -> None:
        """FitzHugh-Rinzel has exact three-way Q16.16 spike-count parity.

        The enrolled schema mirrors the maintained three-state flow: four-stage
        simultaneous RK4 over the cubic fast membrane, linear recovery, and
        ultra-slow modulation equations; no reset; and rising-edge
        ``v >= v_threshold`` crossing detection. Over 3000 steps the hand model,
        schema runner, and emitted Q16.16 RTL produce 7, 8, and 8 crossings at
        ``I=0.4``, ``0.5``, and ``0.6`` respectively. This current band avoids the
        marginal ninth crossing at ``I=0.7``, where fixed-point rounding changes the
        spike count, so the contract states the robust band rather than hiding that
        boundary.
        """
        n_steps = 3000
        hand_spikes = _fitzhugh_rinzel_hand_spike_count(n_steps, current)
        py_spikes = _python_spike_count("fitzhugh_rinzel", n_steps, current)
        vlog_spikes = _verilog_spike_count_q1616("fitzhugh_rinzel", n_steps, current)
        assert hand_spikes == expected_spikes
        assert hand_spikes == py_spikes == vlog_spikes, (
            f"FitzHugh-Rinzel three-way mismatch at I={current}: hand={hand_spikes}, "
            f"schema={py_spikes}, verilog={vlog_spikes}"
        )
