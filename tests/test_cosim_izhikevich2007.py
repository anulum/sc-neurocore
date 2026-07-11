# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Izhikevich 2007 co-simulation contracts

"""Izhikevich 2007 schema, hand-model, and Q16.16 RTL parity contracts."""

from __future__ import annotations

import pytest

from sc_neurocore.neurons.models.izhikevich2007 import Izhikevich2007Neuron
from sc_neurocore.neurons.universal_dsl import UniversalNeuron
from tests.cosim_support import (
    HAS_IVERILOG,
    _izhikevich2007_hand_euler_spike_count,
    _python_spike_count,
    _verilog_spike_count_q1616,
)


class TestTierBModelCosim:
    """WC-A5 Tier-B Izhikevich 2007 enrolment."""

    def test_izhikevich2007_schema_matches_hand_euler_sequence(self) -> None:
        """The schema mirrors the Izhikevich 2007 Euler step law and reset over a sequence.

        The bundled ``izhikevich2007`` schema is the explicit-Euler discretisation of
        ``Izhikevich2007Neuron(integrator="euler")`` — the model also ships an RK4
        default, validated separately through the RK4-emitter path. This three-way
        anchor asserts the schema reproduces the hand model's spike decision *and* both
        state variables over a varied drive, catching any silent drift from the
        canonical publication implementation.
        """
        hand = Izhikevich2007Neuron(integrator="euler")
        schema = UniversalNeuron.from_schema("izhikevich2007")

        for current in (0.0, 200.0, 1000.0, 500.0, 1000.0, 0.0, 700.0, 1500.0):
            assert int(bool(schema.step(I=current))) == hand.step(current)
            assert schema.state["v"] == hand.v
            assert schema.state["u"] == hand.u

    @pytest.mark.skipif(not HAS_IVERILOG, reason="Icarus Verilog not available")
    def test_izhikevich2007_q1616_matches_hand_and_verilog(self) -> None:
        """Izhikevich 2007 (Euler) has exact Q16.16 spike-count parity across all paths.

        The regular-spiking operating point (``I=1000`` pA, 500 steps) fires a partial
        train (8 of 500 steps) after ~57 steps of sub-threshold accumulation, so the
        test exercises multi-step accumulation and threshold-crossing timing rather than
        a saturated every-step spike. ``dt=0.1``, ``k=0.7`` and ``a=0.03`` are not
        exactly representable in Q16.16, so the fixed-point datapath is genuinely
        stressed, yet the polynomial right-hand side and 32-bit word reproduce the float
        spike train exactly across the hand model, the schema runner and the emitted RTL.
        """
        hand_spikes = _izhikevich2007_hand_euler_spike_count(500, 1000.0)
        schema_spikes = _python_spike_count("izhikevich2007", 500, 1000.0)
        verilog_spikes = _verilog_spike_count_q1616("izhikevich2007", 500, 1000.0)

        assert 0 < schema_spikes < 500  # a partial train, neither saturated nor silent
        assert hand_spikes == schema_spikes == verilog_spikes
