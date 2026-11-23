# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLayerConfig from former test_sc_nas_engine.py

"""Focused suite: TestLayerConfig from former test_sc_nas_engine.py."""

from __future__ import annotations

from sc_neurocore.nas.sc_nas_engine import (
    DecorrelationStrategy,
    LayerConfig,
    NeuronType,
)


class TestLayerConfig:
    """Resource costs of one network layer."""

    def test_lut_cost_increases_with_neurons(self) -> None:
        """More neurons cost more LUTs."""
        a = LayerConfig(32, NeuronType.LIF, 256, DecorrelationStrategy.LFSR)
        b = LayerConfig(64, NeuronType.LIF, 256, DecorrelationStrategy.LFSR)
        assert b.lut_cost > a.lut_cost

    def test_ff_cost_increases_with_bitstream_length(self) -> None:
        """A longer bitstream costs more flip-flops."""
        a = LayerConfig(32, NeuronType.LIF, 128, DecorrelationStrategy.LFSR)
        b = LayerConfig(32, NeuronType.LIF, 1024, DecorrelationStrategy.LFSR)
        assert b.ff_cost > a.ff_cost

    def test_power_scales_with_length(self) -> None:
        """A longer bitstream draws more power."""
        a = LayerConfig(32, NeuronType.LIF, 256, DecorrelationStrategy.LFSR)
        b = LayerConfig(32, NeuronType.LIF, 512, DecorrelationStrategy.LFSR)
        assert b.power_cost > a.power_cost

    def test_hh_costlier_than_lif(self) -> None:
        """A Hodgkin-Huxley layer costs more LUTs and power than LIF."""
        lif = LayerConfig(32, NeuronType.LIF, 256, DecorrelationStrategy.LFSR)
        hh = LayerConfig(32, NeuronType.HH, 256, DecorrelationStrategy.LFSR)
        assert hh.lut_cost > lif.lut_cost
        assert hh.power_cost > lif.power_cost

    def test_neuron_type_ordering(self) -> None:
        """LIF, Izhikevich, AdEx and Hodgkin-Huxley rise in cost in that order."""
        costs = {}
        for nt in NeuronType:
            l = LayerConfig(32, nt, 256, DecorrelationStrategy.LFSR)
            costs[nt] = l.lut_cost
        assert costs[NeuronType.LIF] < costs[NeuronType.IZHIKEVICH]
        assert costs[NeuronType.IZHIKEVICH] < costs[NeuronType.ADEX]
        assert costs[NeuronType.ADEX] < costs[NeuronType.HH]

    def test_dsp_cost(self) -> None:
        """LIF uses no DSP and Hodgkin-Huxley four per neuron."""
        lif = LayerConfig(32, NeuronType.LIF, 256, DecorrelationStrategy.LFSR)
        hh = LayerConfig(32, NeuronType.HH, 256, DecorrelationStrategy.LFSR)
        assert lif.dsp_cost == 0
        assert hh.dsp_cost == 32 * 4

    def test_bram_cost(self) -> None:
        """The BRAM cost follows the neuron count and bitstream length."""
        l = LayerConfig(64, NeuronType.LIF, 1024, DecorrelationStrategy.LFSR)
        expected = (64 * 1024) / 8192.0
        assert abs(l.bram_cost_kb - expected) < 0.01
