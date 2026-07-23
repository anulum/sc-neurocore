# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestEvoStatistics from former test_statistics.py

"""Focused suite: TestEvoStatistics from former test_statistics.py."""

from __future__ import annotations

from tests.test_evo_substrate.statistics_support import *  # noqa: F403

class TestEvoStatistics:
    def test_record(self) -> None:
        est = EvoStatisticsTracker()
        est.record(GenerationStats(1, 10, 0.7, 0.5, 0.3))
        est.record(GenerationStats(2, 12, 0.8, 0.6, 0.25))
        assert est.generations_tracked == 2

    def test_trajectory(self) -> None:
        est = EvoStatisticsTracker()
        est.record(GenerationStats(1, 10, 0.5, 0.3, 0.2))
        est.record(GenerationStats(2, 10, 0.8, 0.5, 0.3))
        assert est.fitness_trajectory == [0.5, 0.8]
        assert est.improvement_rate() == pytest.approx(0.3)

    def test_single_record_has_diversity_trajectory_and_zero_improvement(self) -> None:
        est = EvoStatisticsTracker()
        est.record(GenerationStats(1, 10, 0.5, 0.3, 0.2))

        assert est.diversity_trajectory == [0.2]
        assert est.improvement_rate() == 0.0
