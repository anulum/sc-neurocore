# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestReceptiveField from former test_spike_train_stats_extended.py

"""Focused suite: TestReceptiveField from former test_spike_train_stats_extended.py."""

from __future__ import annotations

from tests.spike_train_stats_extended_support import *  # noqa: F403


class TestReceptiveField:
    def test_spike_triggered_covariance(self, poisson_train):
        stim = np.random.default_rng(3).normal(0, 1, poisson_train.size)
        stc = spike_triggered_covariance(stim, poisson_train, window_steps=20)
        assert stc.shape == (20, 20)

    def test_spatial_information(self, poisson_train):
        positions = np.linspace(0, 100, poisson_train.size)
        si = spatial_information(poisson_train, positions)
        assert si >= 0

    def test_place_field_detection(self):
        rng = np.random.default_rng(9)
        n = 5000
        positions = np.linspace(0, 100, n)
        train = (rng.random(n) < 0.005).astype(np.float64)
        train[(positions > 40) & (positions < 60)] += (
            rng.random(int(((positions > 40) & (positions < 60)).sum())) < 0.1
        ).astype(np.float64)
        train = np.clip(train, 0, 1)
        fields = place_field_detection(train, positions, threshold_std=1.5)
        assert isinstance(fields, list)

    def test_tuning_curve(self, poisson_train):
        stim = np.sin(np.linspace(0, 4 * np.pi, poisson_train.size))
        rates, centers = tuning_curve(poisson_train, stim)
        assert rates.size == centers.size
        assert rates.size > 0
