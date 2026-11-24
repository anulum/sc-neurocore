# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDistanceMetrics from former test_spike_train_stats_extended.py

"""Focused suite: TestDistanceMetrics from former test_spike_train_stats_extended.py."""

from __future__ import annotations

from typing import Any

from sc_neurocore.analysis.spike_stats import adaptive_spike_distance
from sc_neurocore.analysis.spike_stats import earth_movers_distance
from sc_neurocore.analysis.spike_stats import generalized_victor_purpura
from sc_neurocore.analysis.spike_stats import hunter_milton_similarity
from sc_neurocore.analysis.spike_stats import isi_profile
from sc_neurocore.analysis.spike_stats import multi_neuron_victor_purpura
from sc_neurocore.analysis.spike_stats import schreiber_similarity
from sc_neurocore.analysis.spike_stats import spike_distance
from sc_neurocore.analysis.spike_stats import spike_distance_matrix
from sc_neurocore.analysis.spike_stats import spike_profile
from sc_neurocore.analysis.spike_stats import spike_sync
from sc_neurocore.analysis.spike_stats import spike_sync_profile
import numpy as np
from tests.spike_train_stats_extended_support import poisson_train as poisson_train
from tests.spike_train_stats_extended_support import spike_times_pair as spike_times_pair
from tests.spike_train_stats_extended_support import two_trains as two_trains


class TestDistanceMetrics:
    """Spike-time distances, similarities and their profiles and matrices."""

    def test_spike_distance_identical(self) -> None:
        """The SPIKE distance of a train to itself is below 0.01."""
        t = np.array([0.1, 0.3, 0.5, 0.7])
        assert spike_distance(t, t) < 0.01

    def test_spike_distance_different(
        self, spike_times_pair: tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]
    ) -> None:
        """The SPIKE distance of two random trains is non-negative."""
        ta, tb = spike_times_pair
        d = spike_distance(ta, tb)
        assert d >= 0

    def test_spike_sync_identical(self) -> None:
        """The spike synchrony of a train with itself is above 0.5."""
        t = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        assert spike_sync(t, t) > 0.5

    def test_spike_sync_profile(
        self, spike_times_pair: tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]
    ) -> None:
        """The spike-synchrony profile has the default 50 bins."""
        ta, tb = spike_times_pair
        prof = spike_sync_profile(ta, tb)
        assert prof.shape == (50,)

    def test_spike_profile(
        self, spike_times_pair: tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]
    ) -> None:
        """The SPIKE-distance profile has the default 50 bins."""
        ta, tb = spike_times_pair
        prof = spike_profile(ta, tb)
        assert prof.shape == (50,)

    def test_isi_profile(
        self, two_trains: tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]
    ) -> None:
        """The ISI-distance profile has the default 50 bins."""
        a, b = two_trains
        prof = isi_profile(a, b)
        assert prof.shape == (50,)

    def test_adaptive_spike_distance(
        self, spike_times_pair: tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]
    ) -> None:
        """The adaptive SPIKE distance of two random trains is non-negative."""
        ta, tb = spike_times_pair
        d = adaptive_spike_distance(ta, tb)
        assert d >= 0

    def test_schreiber_similarity_identical(self, poisson_train: np.ndarray[Any, Any]) -> None:
        """The Schreiber similarity of a train with itself is above 0.99."""
        s = schreiber_similarity(poisson_train, poisson_train)
        assert s > 0.99

    def test_hunter_milton(
        self, spike_times_pair: tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]
    ) -> None:
        """The Hunter-Milton similarity with a 50 ms window lies inside [0, 1]."""
        ta, tb = spike_times_pair
        s = hunter_milton_similarity(ta, tb, dt_max=0.05)
        assert 0.0 <= s <= 1.0

    def test_earth_movers_distance(
        self, spike_times_pair: tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]
    ) -> None:
        """The earth mover's distance of two random trains is non-negative."""
        ta, tb = spike_times_pair
        d = earth_movers_distance(ta, tb)
        assert d >= 0

    def test_multi_neuron_victor_purpura(
        self, spike_times_pair: tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]
    ) -> None:
        """The multi-neuron Victor-Purpura matrix of two trains is 2x2 and symmetric, with a zero self-distance."""
        ta, tb = spike_times_pair
        mat = multi_neuron_victor_purpura([ta, tb])
        assert mat.shape == (2, 2)
        assert mat[0, 0] == 0.0
        assert mat[0, 1] == mat[1, 0]

    def test_generalized_victor_purpura(
        self, spike_times_pair: tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]
    ) -> None:
        """The generalised Victor-Purpura distance of two random trains is non-negative."""
        ta, tb = spike_times_pair
        d = generalized_victor_purpura(ta, tb)
        assert d >= 0

    def test_spike_distance_matrix(
        self, spike_times_pair: tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]
    ) -> None:
        """The SPIKE-distance matrix of two trains is 2x2 and symmetric."""
        ta, tb = spike_times_pair
        mat = spike_distance_matrix([ta, tb])
        assert mat.shape == (2, 2)
        assert mat[0, 1] == mat[1, 0]
