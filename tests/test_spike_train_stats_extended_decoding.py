# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDecoding from former test_spike_train_stats_extended.py

"""Focused suite: TestDecoding from former test_spike_train_stats_extended.py."""

from __future__ import annotations

from tests.spike_train_stats_extended_support import *  # noqa: F403

class TestDecoding:
    def test_bayesian_decode(self):
        tuning = np.array([[10.0, 1.0], [1.0, 10.0], [5.0, 5.0]])
        counts = np.array([9, 2])
        result = bayesian_decode(counts, tuning)
        assert result == 0

    def test_maximum_likelihood_decode(self):
        tuning = np.array([[10.0, 1.0], [1.0, 10.0]])
        counts = np.array([1, 12])
        result = maximum_likelihood_decode(counts, tuning)
        assert result == 1

    def test_linear_discriminant_decode(self):
        rng = np.random.default_rng(8)
        train = np.vstack([rng.normal(0, 1, (20, 3)), rng.normal(3, 1, (20, 3))])
        labels = np.concatenate([np.zeros(20), np.ones(20)])
        test = np.array([3.0, 3.0, 3.0])
        pred = linear_discriminant_decode(train, labels, test)
        assert pred == 1

    def test_naive_bayes_decode(self):
        rng = np.random.default_rng(12)
        train = np.vstack([rng.normal(-2, 0.5, (30, 2)), rng.normal(2, 0.5, (30, 2))])
        labels = np.concatenate([np.zeros(30), np.ones(30)])
        test = np.array([2.0, 2.0])
        pred = naive_bayes_decode(train, labels, test)
        assert pred == 1
