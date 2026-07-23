# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFitnessTrajectory from former test_meta_plasticity.py

"""Focused suite: TestFitnessTrajectory from former test_meta_plasticity.py."""

from __future__ import annotations

from meta_plasticity_support import *  # noqa: F403

class TestFitnessTrajectory:
    def test_improving_trend(self):
        ft = FitnessTrajectory(window=10)
        for i in range(20):
            ft.record(float(i) / 19.0)
        assert ft.trend() > 0
        assert ft.is_improving

    def test_declining_trend(self):
        ft = FitnessTrajectory(window=10)
        for i in range(20):
            ft.record(1.0 - float(i) / 19.0)
        assert ft.trend() < 0

    def test_stagnant(self):
        ft = FitnessTrajectory(window=10)
        for _ in range(20):
            ft.record(0.5)
        assert ft.is_stagnant

    def test_best_ever(self):
        ft = FitnessTrajectory()
        ft.record(0.3)
        ft.record(0.9)
        ft.record(0.5)
        assert ft.best_ever == 0.9

    def test_trend_insufficient_history(self):
        assert FitnessTrajectory().trend() == 0.0

    def test_trend_single_point_window_has_no_slope(self):
        # A window of one collapses the x-axis to a single point with zero
        # variance, so no slope can be fit and the trend is 0.
        ft = FitnessTrajectory(window=1)
        ft.record(1.0)
        ft.record(2.0)
        assert ft.trend() == 0.0

    def test_is_stagnant_false_before_window_fills(self):
        ft = FitnessTrajectory(window=20)
        ft.record(0.5)
        assert ft.is_stagnant is False
