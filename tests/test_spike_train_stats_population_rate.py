# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPopulationRate from former test_spike_train_stats.py

"""Focused suite: TestPopulationRate from former test_spike_train_stats.py."""

from __future__ import annotations

from tests.spike_train_stats_support import *  # noqa: F403


class TestPopulationRate:
    def test_positive(self):
        trains = [_poisson_train(100.0, 0.5, seed=i) for i in range(10)]
        rate = population_rate(trains, sigma_ms=20.0)
        assert rate.size > 0
        assert rate.mean() > 0

    def test_empty(self):
        assert population_rate([]).size == 0
