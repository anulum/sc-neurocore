# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestAnalysisWindow from former test_sc_scope.py

"""Focused suite: TestAnalysisWindow from former test_sc_scope.py."""

from __future__ import annotations

from sc_scope_support import *  # noqa: F403

class TestAnalysisWindow:
    def test_push_and_count(self):
        w = AnalysisWindow(window_size=10)
        for i in range(5):
            w.push(_sample(density=0.5))
        assert w.count == 5

    def test_window_overflow(self):
        w = AnalysisWindow(window_size=4)
        for i in range(10):
            w.push(_sample(density=0.5))
        assert w.count == 4

    def test_mean_density(self):
        w = AnalysisWindow(window_size=100)
        for _ in range(20):
            w.push(_sample(density=1.0))
        assert abs(w.mean_density - 1.0) < 0.01

    def test_std_density(self):
        w = AnalysisWindow(window_size=100)
        for _ in range(20):
            w.push(_sample(density=0.5))
        # All same density → std ≈ 0
        assert w.std_density < 0.01
