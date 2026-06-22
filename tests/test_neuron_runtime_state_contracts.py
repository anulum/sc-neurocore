# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for stochastic neuron runtime state contracts

"""Contracts for stochastic neuron state, reset, refractory, and bitstream paths."""

from __future__ import annotations

import numpy as np
import pytest


def test_sc_izhikevich_initial_state_and_reset_are_consistent() -> None:
    from sc_neurocore.neurons.sc_izhikevich import SCIzhikevichNeuron

    neuron = SCIzhikevichNeuron(seed=42)
    assert neuron.v == neuron.c
    assert neuron.u == neuron.b * neuron.v
    for _ in range(10):
        neuron.step(30.0)

    neuron.reset_state()

    assert neuron.v == neuron.c
    assert {"v", "u"} <= set(neuron.get_state())


def test_sc_izhikevich_spikes_under_strong_drive_and_accepts_noise_source() -> None:
    from sc_neurocore.neurons.sc_izhikevich import SCIzhikevichNeuron

    driven = SCIzhikevichNeuron(seed=1)
    assert 1 in [driven.step(40.0) for _ in range(50)]

    noisy = SCIzhikevichNeuron(noise_std=1.0, seed=42)
    assert noisy.step(5.0) in (0, 1)


def test_stochastic_lif_refractory_resets_voltage_to_rest() -> None:
    from sc_neurocore.neurons.stochastic_lif import StochasticLIFNeuron

    neuron = StochasticLIFNeuron(v_threshold=0.5, refractory_period=3, seed=42, noise_std=0.0)
    for _ in range(50):
        if neuron.step(1.0):
            break

    result = neuron.step(1.0)

    assert result == 0
    assert neuron.v == neuron.v_rest


def test_stochastic_lif_process_bitstream_returns_uint8_trace() -> None:
    from sc_neurocore.neurons.stochastic_lif import StochasticLIFNeuron

    neuron = StochasticLIFNeuron(v_threshold=0.3, seed=42, noise_std=0.0)

    spikes = neuron.process_bitstream(np.ones(50, dtype=np.uint8), input_scale=0.5)

    assert spikes.shape == (50,)
    assert spikes.dtype == np.uint8


def test_fixed_point_lif_reset_state_and_lfsr_seed_reset() -> None:
    from sc_neurocore.neurons.fixed_point_lif import FixedPointBitstreamEncoder
    from sc_neurocore.neurons.fixed_point_lif import FixedPointLFSR
    from sc_neurocore.neurons.fixed_point_lif import FixedPointLIFNeuron

    neuron = FixedPointLIFNeuron()
    neuron.step(26, 128, 200, 0)
    neuron.reset_state()
    assert neuron.v == neuron.v_rest
    assert {"v", "refractory_counter"} <= set(neuron.get_state())

    lfsr = FixedPointLFSR(seed=0xBEEF)
    lfsr.step()
    lfsr.reset(seed=0xCAFE)
    assert lfsr.reg == 0xCAFE

    encoder = FixedPointBitstreamEncoder(seed_init=0xACE1)
    encoder.step(128)
    encoder.reset()


def test_fixed_point_lif_rejects_invalid_fixed_point_format() -> None:
    from sc_neurocore.neurons.fixed_point_lif import FixedPointLIFNeuron

    with pytest.raises(ValueError, match="data_width must be in"):
        FixedPointLIFNeuron(data_width=0)
    with pytest.raises(ValueError, match="fraction must be in"):
        FixedPointLIFNeuron(data_width=8, fraction=8)
    with pytest.raises(ValueError, match="refractory_period must be"):
        FixedPointLIFNeuron(refractory_period=-1)


def test_stochastic_dendritic_neuron_xor_and_reset_contract() -> None:
    from sc_neurocore.neurons.dendritic import StochasticDendriticNeuron

    neuron = StochasticDendriticNeuron()

    assert neuron.step(1.0, 0.0) == 1
    assert neuron.step(0.0, 1.0) == 1
    assert neuron.step(1.0, 1.0) == 0
    assert neuron.step(0.0, 0.0) == 0
    assert "last_current" in neuron.get_state()
    neuron.reset_state()
    assert neuron._last_current == 0.0
