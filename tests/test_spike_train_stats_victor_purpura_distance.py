# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestVictorPurpuraDistance from former test_spike_train_stats.py

"""Focused suite: TestVictorPurpuraDistance from former test_spike_train_stats.py."""

from __future__ import annotations

from tests.spike_train_stats_support import *  # noqa: F403


class TestVictorPurpuraDistance:
    def test_identical_zero(self):
        t = np.array([0.1, 0.2, 0.3])
        assert victor_purpura_distance(t, t) < 1e-10

    def test_empty(self):
        assert victor_purpura_distance(np.array([]), np.array([0.1, 0.2])) == 2.0

    def test_different(self):
        a = np.array([0.1, 0.3, 0.5])
        b = np.array([0.15, 0.35, 0.55])
        d = victor_purpura_distance(a, b, cost_per_s=100.0)
        assert d > 0
