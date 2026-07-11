# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Morris-Lecar co-simulation contracts

"""Morris-Lecar schema, hand-model, and Q16.16 RTL parity contracts."""

from __future__ import annotations

import pytest

from tests.cosim_support import (
    HAS_IVERILOG,
    _morris_lecar_hand_spike_count,
    _python_spike_count,
    _verilog_spike_count_q1616,
)


@pytest.mark.skipif(not HAS_IVERILOG, reason="Icarus Verilog not available")
class TestQ1616Precision:
    """Q16.16 Morris-Lecar co-simulation fidelity."""

    def test_morris_lecar_q1616_parity(self) -> None:
        """Faithful Morris-Lecar co-simulates at exact Q16.16 three-way crossing parity.

        The re-enrolled schema is the genuine Morris-Lecar (1981) calcium-potassium
        relaxation oscillator matching ``MorrisLecarNeuron``'s maintained defaults:
        four-stage RK4, **no reset**, and rising-edge (``v >= v_threshold`` upward
        crossing) spike detection. The earlier schema was ``method="euler"`` with a
        no-op ``[reset]`` (``v -> v``, ``w -> w``) that disabled edge detection, routed
        to the level datapath, and over-counted every above-threshold step; both sides
        over-counted identically so a ~15% tolerance band passed while validating a
        caricature. The faithful schema counts one spike per action potential: at the
        sustained depolarising regime (``I=100``, 3000 steps) the hand model, the schema
        runner and the emitted Q16.16 RTL all report the same seven upward crossings.

        The sigmoidal gating lowers to 256-entry cosh/tanh LUTs, so — unlike the
        polynomial FitzHugh-Nagumo / piecewise-linear McKean oscillators — this is an
        exact **spike-count** parity, not bit-identical state: the hand model (``math``
        transcendentals via ``RK4Solver``) and the schema runner (``numpy``
        transcendentals) diverge at the float level, yet the crossing count is robust to
        that drift across the whole ``I in [90, 110]`` band and the Q16.16 LUT datapath
        reproduces it exactly. (``I=120`` is a knife-edge where a marginal crossing
        splits between the paths; the enrolled point sits safely inside the robust band.)
        """
        current, n_steps = 100.0, 3000
        hand_spikes = _morris_lecar_hand_spike_count(n_steps, current)
        py_spikes = _python_spike_count("morris_lecar", n_steps, current)
        vlog_spikes = _verilog_spike_count_q1616("morris_lecar", n_steps, current)
        assert 1 < py_spikes < n_steps  # a sustained relaxation train, not saturated
        assert hand_spikes == py_spikes == vlog_spikes, (
            f"Morris-Lecar three-way mismatch: hand={hand_spikes}, "
            f"schema={py_spikes}, verilog={vlog_spikes}"
        )
