# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPhiFromSpikeTrains from former test_phi_estimation.py

"""Focused suite: TestPhiFromSpikeTrains from former test_phi_estimation.py."""

from __future__ import annotations

from tests.phi_estimation_support import *  # noqa: F403

class TestPhiFromSpikeTrains:
    def test_spike_trains_integration(self) -> None:
        rng = np.random.RandomState(42)
        n_neurons, n_steps = 4, 1000
        shared = rng.random(n_steps) < 0.3
        spikes = np.zeros((n_neurons, n_steps), dtype=np.uint8)
        for i in range(n_neurons):
            noise = rng.random(n_steps) < 0.1
            spikes[i] = np.bitwise_xor(shared.astype(np.uint8), noise.astype(np.uint8))
        assert phi_from_spike_trains(spikes, bin_size=10, tau=1, backend="python") >= 0.0

    def test_random_spikes_low_phi(self) -> None:
        rng = np.random.RandomState(42)
        spikes = (rng.random((4, 500)) < 0.3).astype(np.uint8)
        assert phi_from_spike_trains(spikes, bin_size=10, tau=1, backend="python") < 1.0

    def test_too_short_returns_zero(self) -> None:
        spikes = np.zeros((3, 10), dtype=np.uint8)
        assert phi_from_spike_trains(spikes, bin_size=5, tau=1) == 0.0
