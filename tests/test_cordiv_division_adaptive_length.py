# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestAdaptiveLength from former test_cordiv_division.py

"""Focused suite: TestAdaptiveLength from former test_cordiv_division.py."""

from __future__ import annotations

from tests.cordiv_division_support import *  # noqa: F403


class TestAdaptiveLength:
    def test_returns_positive_int(self):
        L = adaptive_length(0.5, epsilon=0.01, confidence=0.95)
        assert isinstance(L, int)
        assert L > 0

    def test_tighter_bound_needs_longer(self):
        L1 = adaptive_length(0.5, epsilon=0.1, confidence=0.95)
        L2 = adaptive_length(0.5, epsilon=0.01, confidence=0.95)
        assert L2 > L1

    def test_higher_confidence_needs_longer(self):
        L1 = adaptive_length(0.5, epsilon=0.05, confidence=0.90)
        L2 = adaptive_length(0.5, epsilon=0.05, confidence=0.99)
        assert L2 > L1

    def test_respects_min_length(self):
        L = adaptive_length(0.5, epsilon=0.5, confidence=0.5, min_length=64)
        assert L >= 64

    def test_respects_max_length(self):
        L = adaptive_length(0.5, epsilon=0.001, confidence=0.999, max_length=1024)
        assert L <= 1024

    def test_hoeffding_formula(self):
        """L >= ln(2/delta) / (2*eps^2) for Hoeffding bound."""
        eps = 0.05
        delta = 0.05  # 95% confidence
        expected_min = np.log(2.0 / delta) / (2.0 * eps**2)
        L = adaptive_length(0.5, epsilon=eps, confidence=0.95, max_length=100000)
        assert int(expected_min) - 1 <= L, f"L={L} < Hoeffding {expected_min:.0f}"
