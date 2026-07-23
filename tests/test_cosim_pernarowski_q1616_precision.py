# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestQ1616Precision from former test_cosim_pernarowski.py

"""Focused suite: TestQ1616Precision from former test_cosim_pernarowski.py."""

from __future__ import annotations

from tests.cosim_pernarowski_support import *  # noqa: F403

@pytest.mark.skipif(not HAS_IVERILOG, reason="Icarus Verilog not available")
class TestQ1616Precision:
    """Q16.16 Pernarowski co-simulation fidelity."""

    @pytest.mark.parametrize(
        "current",
        (-0.1, 0.0, 0.1, 0.2),
        ids=("I=-0.1", "I=0.0", "I=0.1", "I=0.2"),
    )
    def test_pernarowski_q1616_parity(self, current: float) -> None:
        """Pernarowski has exact three-way Q16.16 spike-count parity.

        The enrolled schema mirrors the maintained three-state beta-cell flow:
        simultaneous four-stage RK4 over the cubic fast coordinate and two
        separated slow variables, rising-edge ``v >= v_threshold`` detection,
        and no reset. The oscillator is autonomous, so input current shifts the
        trajectory rather than gating a silent/single/train transition. At each
        enrolled point from ``I=-0.1`` through ``I=0.2``, the hand model, schema
        runner, and emitted Q16.16 RTL report 17 crossings over 5,000 steps.
        """
        n_steps = 5000
        hand_spikes = _pernarowski_hand_spike_count(n_steps, current)
        py_spikes = _python_spike_count("pernarowski", n_steps, current)
        vlog_spikes = _verilog_spike_count_q1616("pernarowski", n_steps, current)
        assert 1 < hand_spikes < n_steps
        assert hand_spikes == py_spikes == vlog_spikes == 17, (
            f"Pernarowski three-way mismatch at I={current}: hand={hand_spikes}, "
            f"schema={py_spikes}, verilog={vlog_spikes}"
        )
