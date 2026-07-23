# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSignificanceBootstrap from former test_spike_train_stats.py

"""Focused suite: TestSignificanceBootstrap from former test_spike_train_stats.py."""

from __future__ import annotations

from tests.spike_train_stats_support import *  # noqa: F403

class TestSignificanceBootstrap:
    def test_returns_pvalue(self):
        a = _poisson_train(100.0, 0.5, seed=1)
        b = _poisson_train(100.0, 0.5, seed=2)

        def stat(x, y):
            return abs(x.mean() - y.mean())

        obs, pval = significance_bootstrap(stat, a, b, n_surrogates=50, seed=42)
        assert 0.0 <= pval <= 1.0
        assert obs >= 0
