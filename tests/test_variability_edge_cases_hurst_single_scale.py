# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestHurstSingleScale from former test_variability_edge_cases.py

"""Focused suite: TestHurstSingleScale from former test_variability_edge_cases.py."""

from __future__ import annotations

from tests.variability_edge_cases_support import *  # noqa: F403


class TestHurstSingleScale:
    """When only one DFA scale fits the train length, the log-log fit has too
    few points and the Hurst exponent is undefined."""

    def test_single_usable_scale_returns_nan(self):
        # n == 4*min_window admits exactly one scale (s=min_window); the next
        # 1.5x step exceeds n//4, leaving a single (log s, log F) point.
        train = np.zeros(40, dtype=np.int8)
        train[::4] = 1
        assert np.isnan(hurst_exponent(train, min_window=10))
