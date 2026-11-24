# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSynchrony from former test_spike_train_stats_extended.py

"""Focused suite: TestSynchrony from former test_spike_train_stats_extended.py."""

from __future__ import annotations

from typing import Any

from sc_neurocore.analysis.spike_stats import autocorrelation_time
from sc_neurocore.analysis.spike_stats import coincidence_index
from sc_neurocore.analysis.spike_stats import covariance_matrix
from sc_neurocore.analysis.spike_stats import joint_psth
from sc_neurocore.analysis.spike_stats import noise_correlation
from sc_neurocore.analysis.spike_stats import signal_correlation
from sc_neurocore.analysis.spike_stats import spike_count_covariance
from sc_neurocore.analysis.spike_stats import spike_time_tiling_coefficient
import numpy as np
from tests.spike_train_stats_extended_support import poisson_train as poisson_train
from tests.spike_train_stats_extended_support import population as population
from tests.spike_train_stats_extended_support import two_trains as two_trains


class TestSynchrony:
    """Synchrony, correlation and coincidence measures."""

    def test_sttc_identical(self, poisson_train: np.ndarray[Any, Any]) -> None:
        """The STTC of a train with itself is above 0.5."""
        val = spike_time_tiling_coefficient(poisson_train, poisson_train)
        assert val > 0.5

    def test_sttc_independent(
        self, two_trains: tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]
    ) -> None:
        """The STTC of two independent trains lies inside [-1, 1]."""
        a, b = two_trains
        val = spike_time_tiling_coefficient(a, b)
        assert -1.0 <= val <= 1.0

    def test_sttc_silent_train_is_zero(self) -> None:
        """The STTC of a silent and an active train is zero."""
        # With no spikes in one train the tiling coefficient is undefined and
        # collapses to zero rather than indexing an empty spike-time array.
        silent = np.zeros(200, dtype=np.float64)
        active = np.zeros(200, dtype=np.float64)
        active[::20] = 1.0
        assert spike_time_tiling_coefficient(silent, active) == 0.0

    def test_covariance_matrix(self, population: list[np.ndarray[Any, Any]]) -> None:
        """The covariance matrix of five trains has five rows."""
        cov = covariance_matrix(population)
        assert cov.shape[0] == 5

    def test_autocorrelation_time(self, poisson_train: np.ndarray[Any, Any]) -> None:
        """The autocorrelation time of a Poisson train is non-negative."""
        tau = autocorrelation_time(poisson_train)
        assert tau >= 0

    def test_noise_correlation(self, population: list[np.ndarray[Any, Any]]) -> None:
        """The noise correlation of five trains is 5x5 with ones on the diagonal."""
        nc = noise_correlation(population)
        assert nc.shape == (5, 5)
        assert np.allclose(np.diag(nc), 1.0)

    def test_signal_correlation(self, population: list[np.ndarray[Any, Any]]) -> None:
        """The signal correlation of five trains is 5x5."""
        sc = signal_correlation(population)
        assert sc.shape == (5, 5)

    def test_spike_count_covariance(self, population: list[np.ndarray[Any, Any]]) -> None:
        """The spike-count covariance of five trains has five rows."""
        cov = spike_count_covariance(population)
        assert cov.shape[0] == 5

    def test_joint_psth(
        self, two_trains: tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]
    ) -> None:
        """The joint PSTH of two trains is two-dimensional."""
        a, b = two_trains
        jp = joint_psth(a, b)
        assert jp.ndim == 2

    def test_coincidence_index(
        self, two_trains: tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]
    ) -> None:
        """The coincidence index of two random trains is finite."""
        a, b = two_trains
        ci = coincidence_index(a, b)
        assert np.isfinite(ci)
