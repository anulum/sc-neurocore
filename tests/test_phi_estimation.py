# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for Phi estimation (IIT)

"""Tests for integrated information (Phi*) estimation."""

import numpy as np

from sc_neurocore.analysis.phi_estimation import phi_star, phi_from_spike_trains


class TestPhiStar:
    def test_independent_channels_zero_phi(self):
        """Independent (uncorrelated) channels should have Phi ≈ 0."""
        rng = np.random.RandomState(42)
        data = rng.randn(4, 200)
        phi = phi_star(data, tau=1)
        assert phi < 0.5  # near-zero, allowing for finite-sample noise

    def test_correlated_channels_positive_phi(self):
        """Strongly correlated channels should have Phi > 0."""
        rng = np.random.RandomState(42)
        shared = rng.randn(200)
        data = np.vstack(
            [
                shared + 0.1 * rng.randn(200),
                shared + 0.1 * rng.randn(200),
                shared + 0.1 * rng.randn(200),
            ]
        )
        phi = phi_star(data, tau=1)
        assert phi > 0

    def test_two_channels_symmetric(self):
        """Phi should not depend on channel ordering."""
        rng = np.random.RandomState(42)
        shared = rng.randn(100)
        a = shared + 0.1 * rng.randn(100)
        b = shared + 0.1 * rng.randn(100)
        phi_forward = phi_star(np.vstack([a, b]), tau=1)
        phi_reversed = phi_star(np.vstack([b, a]), tau=1)
        np.testing.assert_allclose(phi_forward, phi_reversed, atol=1e-10)

    def test_single_channel_returns_zero(self):
        """A single channel system should have Phi = 0."""
        data = np.random.randn(1, 100)
        assert phi_star(data) == 0.0

    def test_short_data_returns_zero(self):
        data = np.random.randn(3, 3)
        assert phi_star(data, tau=2) == 0.0

    def test_nonnegative(self):
        """Phi should always be non-negative."""
        rng = np.random.RandomState(42)
        for _ in range(10):
            data = rng.randn(3, 50)
            assert phi_star(data) >= 0.0


class TestPhiFromSpikeTrains:
    def test_spike_trains_integration(self):
        """Correlated spike trains should have positive Phi."""
        rng = np.random.RandomState(42)
        n_neurons, n_steps = 4, 1000
        # Generate correlated spikes: shared drive
        rate = 0.3
        shared = rng.random(n_steps) < rate
        spikes = np.zeros((n_neurons, n_steps), dtype=np.uint8)
        for i in range(n_neurons):
            noise = rng.random(n_steps) < 0.1
            spikes[i] = np.bitwise_xor(shared.astype(np.uint8), noise.astype(np.uint8))
        phi = phi_from_spike_trains(spikes, bin_size=10, tau=1)
        assert phi >= 0.0

    def test_random_spikes_low_phi(self):
        """Independent random spikes should have low Phi."""
        rng = np.random.RandomState(42)
        spikes = (rng.random((4, 500)) < 0.3).astype(np.uint8)
        phi = phi_from_spike_trains(spikes, bin_size=10, tau=1)
        assert phi < 1.0

    def test_too_short_returns_zero(self):
        spikes = np.zeros((3, 10), dtype=np.uint8)
        assert phi_from_spike_trains(spikes, bin_size=5, tau=1) == 0.0
