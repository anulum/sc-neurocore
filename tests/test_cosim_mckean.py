# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — McKean co-simulation contracts

"""McKean Q16.16 parity contracts."""

from __future__ import annotations

import pytest

from tests.cosim_support import (
    HAS_IVERILOG,
    _mckean_hand_spike_count,
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

    def test_mckean_q1616_parity(self) -> None:
        """Faithful McKean co-simulates at exact Q16.16 three-way parity.

        The McKean (1970) piecewise-linear FitzHugh-Nagumo caricature replaces the
        cubic nullcline with ``f(v) = min(max(-v, v - a), 1 - v)``; the bundled schema
        is RK4, no reset, rising-edge (``v >= v_peak`` upward crossing) detection,
        matching ``McKeanNeuron``. The min/max branch selection is exact arithmetic (a
        fixed-point comparison + select, no look-up table), so at the sustained
        relaxation-oscillation operating point (``epsilon=0.2``, ``gamma=0.5``,
        ``I=0.6``) the hand model, the schema runner and the emitted Q16.16 RTL all
        report the same 16-crossing train over 3000 steps, bit-exactly. (The default
        hand-model regime ``epsilon=0.01`` is a single-transient knife-edge; the
        enrolled regime is a robust limit cycle whose crossings survive fixed-point
        rounding.)
        """
        current, n_steps = 0.6, 3000
        hand_spikes = _mckean_hand_spike_count(n_steps, current)
        py_spikes = _python_spike_count("mckean", n_steps, current)
        vlog_spikes = _verilog_spike_count_q1616("mckean", n_steps, current)
        assert 1 < py_spikes < n_steps  # a sustained oscillation train, not saturated
        assert hand_spikes == py_spikes == vlog_spikes, (
            f"McKean three-way mismatch: hand={hand_spikes}, "
            f"schema={py_spikes}, verilog={vlog_spikes}"
        )
