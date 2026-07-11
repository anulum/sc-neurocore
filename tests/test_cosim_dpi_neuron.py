# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — DPI neuron co-simulation contracts

"""DPI schema, hand-model, and Q16.16 RTL parity contracts."""

from __future__ import annotations

import pytest

from sc_neurocore.neurons.models.dpi_neuron import DPINeuron
from sc_neurocore.neurons.universal_dsl import UniversalNeuron
from tests.cosim_support import (
    HAS_IVERILOG,
    _dpi_neuron_hand_spike_count,
    _python_spike_count,
    _verilog_spike_count_q1616,
)


class TestTierBModelCosim:
    """WC-A5 Tier-B DPI neuron enrolment."""

    def test_dpi_neuron_schema_matches_hand_euler_sequence(self) -> None:
        """The schema mirrors the DPI current-mode Euler step law and reset over a sequence.

        The bundled ``dpi_neuron`` schema is the explicit-Euler discretisation of the
        DYNAP-SE differential-pair integrator (``DPINeuron``). Because the drive is
        non-negative the source model's ``max(i_mem, 0)`` current rectification never
        engages, so this three-way anchor asserts the schema reproduces the hand model's
        spike decision *and* the membrane current at every step of a varied non-negative
        drive, catching any silent drift from the published circuit model.
        """
        hand = DPINeuron()
        schema = UniversalNeuron.from_schema("dpi_neuron")

        for current in (0.0, 1.5, 3.0, 0.5, 5.0, 0.0, 2.0):
            assert int(bool(schema.step(I=current))) == hand.step(current)
            assert schema.state["i_mem"] == hand.i_mem

    @pytest.mark.skipif(not HAS_IVERILOG, reason="Icarus Verilog not available")
    def test_dpi_neuron_q1616_matches_hand_and_verilog(self) -> None:
        """DPI (Euler) has exact Q16.16 spike-count parity across all three paths.

        The subthreshold operating point (``I=1.5`` nA, 200 steps) fires a partial train
        (9 of 200 steps) after ~22 steps of leaky accumulation, so the test exercises the
        current-mode integrator's asymptotic threshold approach rather than a saturated
        every-step spike. ``i_leak=0.01`` and the ``1/tau=1/20`` membrane gain are not
        exactly representable in Q16.16, so the fixed-point datapath is genuinely stressed,
        yet the linear right-hand side and 32-bit word reproduce the float spike train
        exactly across the hand model, the schema runner and the emitted RTL.
        """
        hand_spikes = _dpi_neuron_hand_spike_count(200, 1.5)
        schema_spikes = _python_spike_count("dpi_neuron", 200, 1.5)
        verilog_spikes = _verilog_spike_count_q1616("dpi_neuron", 200, 1.5)

        assert 0 < schema_spikes < 200  # a partial train, neither saturated nor silent
        assert hand_spikes == schema_spikes == verilog_spikes
