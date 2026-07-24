# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCrossCorrelation from former test_spike_train_stats.py

"""Focused suite: TestCrossCorrelation from former test_spike_train_stats.py."""

from __future__ import annotations

from tests.spike_train_stats_support import *  # noqa: F403


class TestCrossCorrelation:
    def test_autocorrelation_peak_at_zero(self):
        train = _poisson_train(100.0, 1.0)
        cc, lags = cross_correlation(train, train, max_lag_ms=20.0)
        zero_idx = len(lags) // 2
        assert cc[zero_idx] == cc.max()

    def test_independent_low_correlation(self):
        a = _poisson_train(100.0, 1.0, seed=1)
        b = _poisson_train(100.0, 1.0, seed=2)
        cc, _ = cross_correlation(a, b, max_lag_ms=10.0)
        assert np.abs(cc).max() < 0.3

    def test_silent_train_returns_zero_correlogram(self) -> None:
        # A silent train has zero variance, so the normaliser is zero and the
        # correlogram is returned flat rather than dividing by zero.
        silent = np.zeros(200, dtype=np.float64)
        cc, lags = cross_correlation(silent, silent, max_lag_ms=10.0)
        assert np.all(cc == 0.0)
        assert cc.size == lags.size
