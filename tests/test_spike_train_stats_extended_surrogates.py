# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSurrogates from former test_spike_train_stats_extended.py

"""Focused suite: TestSurrogates from former test_spike_train_stats_extended.py."""

from __future__ import annotations

from tests.spike_train_stats_extended_support import *  # noqa: F403


class TestSurrogates:
    def test_homogeneous_poisson(self):
        t = homogeneous_poisson(20.0, 1.0)
        assert t.size == 1000
        rate = t.sum() / 1.0
        assert 5 < rate < 50

    def test_inhomogeneous_poisson(self):
        t = inhomogeneous_poisson(lambda x: 20.0 + 10.0 * np.sin(2 * np.pi * x), 1.0)
        assert t.size == 1000

    def test_gamma_process_shape1(self):
        t = gamma_process(20.0, 1.0, 1.0)
        rate = t.sum()
        assert 5 < rate < 50

    def test_gamma_process_regular(self):
        t = gamma_process(20.0, 10.0, 1.0)
        assert t.sum() > 0

    def test_compound_poisson(self):
        t = compound_poisson_process(10.0, 3.0, 1.0)
        assert t.sum() > 0

    def test_surrogate_joint_isi(self, poisson_train):
        surr = surrogate_joint_isi(poisson_train)
        assert surr.size == poisson_train.size
        assert surr.sum() > 0

    def test_surrogate_bin_shuffling(self, poisson_train):
        surr = surrogate_bin_shuffling(poisson_train)
        assert surr.sum() == poisson_train.sum()

    def test_surrogate_spike_train_shifting(self, poisson_train):
        surr = surrogate_spike_train_shifting(poisson_train)
        assert surr.sum() == poisson_train.sum()

    def test_spike_directionality(self):
        ta = np.array([0.1, 0.2, 0.3, 0.4])
        tb = np.array([0.15, 0.25, 0.35, 0.45])
        d = spike_directionality(ta, tb)
        assert -1.0 <= d <= 1.0

    def test_spike_train_order(self, spike_times_pair):
        ta, tb = spike_times_pair
        mat = spike_train_order([ta, tb])
        assert mat.shape == (2, 2)
        assert np.isclose(mat[0, 1], -mat[1, 0])
