# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCultureHealth from former test_analysis.py

"""Focused suite: TestCultureHealth from former test_analysis.py."""

from __future__ import annotations

from tests.test_bioware.analysis_support import *  # noqa: F403


class TestCultureHealth:
    def test_healthy_culture(self) -> None:
        ch = CultureHealth(min_active_channels=3)
        counts = np.array([10, 15, 20, 5, 8, 0, 0, 0])
        result = ch.assess(counts, duration_s=1.0)
        assert result["is_viable"] is True

    def test_dead_culture(self) -> None:
        ch = CultureHealth(min_active_channels=5)
        counts = np.zeros(60)
        result = ch.assess(counts, duration_s=1.0)
        assert result["health_score"] == 0.0
        assert result["is_viable"] is False

    def test_bursting_detection(self) -> None:
        ch = CultureHealth(burst_threshold_hz=50.0)
        counts = np.array([100, 200, 5, 3])
        result = ch.assess(counts, duration_s=1.0)
        assert result["bursting_channels"] == 2

    def test_excessive_firing_rate_caps_health(self) -> None:
        # A mean rate above the hyperactivity ceiling scales the health score
        # down rather than leaving it at 1.0.
        ch = CultureHealth(min_active_channels=1, max_firing_rate_hz=10.0)
        counts = np.full(8, 1000.0)
        result = ch.assess(counts, duration_s=1.0)
        assert result["health_score"] < 1.0
