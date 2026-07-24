# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestNonstationarity from former test_spike_train_stats_extended.py

"""Focused suite: TestNonstationarity from former test_spike_train_stats_extended.py."""

from __future__ import annotations

from tests.spike_train_stats_extended_support import *  # noqa: F403


class TestNonstationarity:
    def test_change_point_detection(self):
        t = np.zeros(2000)
        rng = np.random.default_rng(11)
        t[:1000] = (rng.random(1000) < 0.01).astype(np.float64)
        t[1000:] = (rng.random(1000) < 0.1).astype(np.float64)
        cps = change_point_detection(t, bin_size=50, threshold=3.0)
        assert isinstance(cps, list)
        assert len(cps) > 0

    def test_cubic_higher_order(self, poisson_train):
        c3 = cubic_higher_order(poisson_train, max_lag=10)
        assert c3.shape == (10, 10)
