# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSynchrony from former test_spike_train_stats_extended.py

"""Focused suite: TestSynchrony from former test_spike_train_stats_extended.py."""

from __future__ import annotations

from tests.spike_train_stats_extended_support import *  # noqa: F403


class TestSynchrony:
    def test_sttc_identical(self, poisson_train):
        val = spike_time_tiling_coefficient(poisson_train, poisson_train)
        assert val > 0.5

    def test_sttc_independent(self, two_trains):
        a, b = two_trains
        val = spike_time_tiling_coefficient(a, b)
        assert -1.0 <= val <= 1.0

    def test_sttc_silent_train_is_zero(self) -> None:
        # With no spikes in one train the tiling coefficient is undefined and
        # collapses to zero rather than indexing an empty spike-time array.
        silent = np.zeros(200, dtype=np.float64)
        active = np.zeros(200, dtype=np.float64)
        active[::20] = 1.0
        assert spike_time_tiling_coefficient(silent, active) == 0.0

    def test_covariance_matrix(self, population):
        cov = covariance_matrix(population)
        assert cov.shape[0] == 5

    def test_autocorrelation_time(self, poisson_train):
        tau = autocorrelation_time(poisson_train)
        assert tau >= 0

    def test_noise_correlation(self, population):
        nc = noise_correlation(population)
        assert nc.shape == (5, 5)
        assert np.allclose(np.diag(nc), 1.0)

    def test_signal_correlation(self, population):
        sc = signal_correlation(population)
        assert sc.shape == (5, 5)

    def test_spike_count_covariance(self, population):
        cov = spike_count_covariance(population)
        assert cov.shape[0] == 5

    def test_joint_psth(self, two_trains):
        a, b = two_trains
        jp = joint_psth(a, b)
        assert jp.ndim == 2

    def test_coincidence_index(self, two_trains):
        a, b = two_trains
        ci = coincidence_index(a, b)
        assert np.isfinite(ci)
