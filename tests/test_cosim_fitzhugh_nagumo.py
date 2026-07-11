# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — FitzHugh-Nagumo co-simulation contracts

"""FitzHugh-Nagumo Q16.16 parity contracts."""

from __future__ import annotations

import pytest

from tests.cosim_support import (
    HAS_IVERILOG,
    _fitzhugh_nagumo_hand_spike_count,
    _python_spike_count,
    _verilog_spike_count_q1616,
)


@pytest.mark.skipif(not HAS_IVERILOG, reason="Icarus Verilog not available")
class TestQ1616Precision:
    """Q16.16 precision mode: 16 integer + 16 fractional bits (32-bit).

    Q16.16 combines Q8.8's wide integer range [-32768, +32767] with
    1/65536 ≈ 0.000015 resolution. This is the "gold standard" for
    hardware neuron fidelity, suitable for all model dynamics.
    """

    def test_fitzhugh_nagumo_q1616_parity(self) -> None:
        """Faithful FitzHugh-Nagumo co-simulates at exact Q16.16 three-way parity.

        The re-enrolled schema is the genuine FitzHugh (1961) relaxation oscillator:
        four-stage RK4, **no reset**, and rising-edge (``v >= v_threshold`` upward
        crossing) spike detection matching ``FitzHughNagumoNeuron`` — the cube is
        ``v * v * v`` (exact IEEE multiplication). Over 3000 steps at ``I=0.5`` the
        hand model, the schema runner and the emitted Q16.16 RTL all report the same
        sustained partial train (eight crossings), a repetitive train that exercises
        the ``_thr_prev`` edge re-arming rather than a single event. The right-hand
        side is polynomial (no look-up table), so the fixed-point parity is bit-exact,
        not a tolerance band. This supersedes the earlier Euler+reset caricature
        (``I=0.8``, 7 of 300) that only agreed because both sides shared the same
        unfaithful reset dynamics.
        """
        current, n_steps = 0.5, 3000
        hand_spikes = _fitzhugh_nagumo_hand_spike_count(n_steps, current)
        py_spikes = _python_spike_count("fitzhugh_nagumo", n_steps, current)
        vlog_spikes = _verilog_spike_count_q1616("fitzhugh_nagumo", n_steps, current)
        assert 1 < py_spikes < n_steps  # a repetitive partial train, not saturated
        assert hand_spikes == py_spikes == vlog_spikes, (
            f"FitzHugh-Nagumo three-way mismatch: hand={hand_spikes}, "
            f"schema={py_spikes}, verilog={vlog_spikes}"
        )
