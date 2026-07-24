# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFiringRatesToSCProbs from former test_spikeinterface_adapter.py

"""Focused suite: TestFiringRatesToSCProbs from former test_spikeinterface_adapter.py."""

from __future__ import annotations

from tests.spikeinterface_adapter_support import *  # noqa: F403


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
