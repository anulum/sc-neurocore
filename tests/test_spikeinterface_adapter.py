# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for SpikeInterface adapter

"""Tests for spike train → bitstream/population conversion."""

import numpy as np

from sc_neurocore.adapters.spikeinterface import (
    firing_rates_to_sc_probs,
    spike_trains_to_bitstreams,
    spike_trains_to_population_input,
)


class TestSpikeTrainsToBitstreams:
    def test_output_shape(self):
        spikes = {0: np.array([10.0, 20.0]), 1: np.array([15.0])}
        mat = spike_trains_to_bitstreams(spikes, duration_ms=50.0, dt=1.0)
        assert mat.shape == (2, 50)

    def test_binary_output(self):
        spikes = {0: np.array([5.0, 10.0])}
        mat = spike_trains_to_bitstreams(spikes, duration_ms=20.0, dt=1.0)
        assert set(np.unique(mat)).issubset({0, 1})

    def test_spike_at_correct_bin(self):
        spikes = {0: np.array([5.0])}
        mat = spike_trains_to_bitstreams(spikes, duration_ms=10.0, dt=1.0)
        assert mat[0, 5] == 1
        assert mat[0, 0] == 0

    def test_empty_unit(self):
        spikes = {0: np.array([]), 1: np.array([3.0])}
        mat = spike_trains_to_bitstreams(spikes, duration_ms=10.0, dt=1.0)
        assert mat[0].sum() == 0
        assert mat[1].sum() == 1

    def test_dt_binning(self):
        spikes = {0: np.array([0.5, 1.5, 2.5])}
        mat = spike_trains_to_bitstreams(spikes, duration_ms=5.0, dt=2.0)
        assert mat.shape == (1, 3)  # ceil(5/2) = 3 bins


class TestSpikeTrainsToPopulationInput:
    def test_output_shape(self):
        spikes = {0: np.array([1.0, 3.0]), 1: np.array([2.0])}
        inp = spike_trains_to_population_input(spikes, duration_ms=5.0, dt=1.0)
        assert inp.shape == (5, 2)  # (n_timesteps, n_units)

    def test_values_are_float(self):
        spikes = {0: np.array([1.0])}
        inp = spike_trains_to_population_input(spikes, duration_ms=3.0, dt=1.0)
        assert inp.dtype == np.float64


class TestFiringRatesToSCProbs:
    def test_output_shape(self):
        spikes = {0: np.array([1.0, 2.0]), 1: np.array([3.0])}
        probs = firing_rates_to_sc_probs(spikes, duration_ms=100.0)
        assert probs.shape == (2,)

    def test_bounded(self):
        spikes = {0: np.linspace(0, 1000, 200)}  # 200 spikes in 1s = 200 Hz
        probs = firing_rates_to_sc_probs(spikes, duration_ms=1000.0, max_rate_hz=100.0)
        assert 0.0 <= probs[0] <= 1.0

    def test_zero_spikes(self):
        spikes = {0: np.array([])}
        probs = firing_rates_to_sc_probs(spikes, duration_ms=1000.0)
        assert probs[0] == 0.0

    def test_rate_scaling(self):
        spikes = {0: np.arange(0, 1000, 20)}  # 50 spikes in 1s = 50 Hz
        probs = firing_rates_to_sc_probs(spikes, duration_ms=1000.0, max_rate_hz=100.0)
        np.testing.assert_allclose(probs[0], 0.5, atol=0.05)
