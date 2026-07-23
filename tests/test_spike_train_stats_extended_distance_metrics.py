# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDistanceMetrics from former test_spike_train_stats_extended.py

"""Focused suite: TestDistanceMetrics from former test_spike_train_stats_extended.py."""

from __future__ import annotations

from tests.spike_train_stats_extended_support import *  # noqa: F403

class TestDistanceMetrics:
    def test_spike_distance_identical(self):
        t = np.array([0.1, 0.3, 0.5, 0.7])
        assert spike_distance(t, t) < 0.01

    def test_spike_distance_different(self, spike_times_pair):
        ta, tb = spike_times_pair
        d = spike_distance(ta, tb)
        assert d >= 0

    def test_spike_sync_identical(self):
        t = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        assert spike_sync(t, t) > 0.5

    def test_spike_sync_profile(self, spike_times_pair):
        ta, tb = spike_times_pair
        prof = spike_sync_profile(ta, tb)
        assert prof.shape == (50,)

    def test_spike_profile(self, spike_times_pair):
        ta, tb = spike_times_pair
        prof = spike_profile(ta, tb)
        assert prof.shape == (50,)

    def test_isi_profile(self, two_trains):
        a, b = two_trains
        prof = isi_profile(a, b)
        assert prof.shape == (50,)

    def test_adaptive_spike_distance(self, spike_times_pair):
        ta, tb = spike_times_pair
        d = adaptive_spike_distance(ta, tb)
        assert d >= 0

    def test_schreiber_similarity_identical(self, poisson_train):
        s = schreiber_similarity(poisson_train, poisson_train)
        assert s > 0.99

    def test_hunter_milton(self, spike_times_pair):
        ta, tb = spike_times_pair
        s = hunter_milton_similarity(ta, tb, dt_max=0.05)
        assert 0.0 <= s <= 1.0

    def test_earth_movers_distance(self, spike_times_pair):
        ta, tb = spike_times_pair
        d = earth_movers_distance(ta, tb)
        assert d >= 0

    def test_multi_neuron_victor_purpura(self, spike_times_pair):
        ta, tb = spike_times_pair
        mat = multi_neuron_victor_purpura([ta, tb])
        assert mat.shape == (2, 2)
        assert mat[0, 0] == 0.0
        assert mat[0, 1] == mat[1, 0]

    def test_generalized_victor_purpura(self, spike_times_pair):
        ta, tb = spike_times_pair
        d = generalized_victor_purpura(ta, tb)
        assert d >= 0

    def test_spike_distance_matrix(self, spike_times_pair):
        ta, tb = spike_times_pair
        mat = spike_distance_matrix([ta, tb])
        assert mat.shape == (2, 2)
        assert mat[0, 1] == mat[1, 0]
