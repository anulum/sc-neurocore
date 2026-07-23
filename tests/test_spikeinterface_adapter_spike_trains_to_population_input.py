# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSpikeTrainsToPopulationInput from former test_spikeinterface_adapter.py

"""Focused suite: TestSpikeTrainsToPopulationInput from former test_spikeinterface_adapter.py."""

from __future__ import annotations

from tests.spikeinterface_adapter_support import *  # noqa: F403

class TestSpikeTrainsToPopulationInput:
    def test_output_shape(self):
        spikes = {0: np.array([1.0, 3.0]), 1: np.array([2.0])}
        inp = spike_trains_to_population_input(spikes, duration_ms=5.0, dt=1.0)
        assert inp.shape == (5, 2)  # (n_timesteps, n_units)

    def test_values_are_float(self):
        spikes = {0: np.array([1.0])}
        inp = spike_trains_to_population_input(spikes, duration_ms=3.0, dt=1.0)
        assert inp.dtype == np.float64
