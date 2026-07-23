# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPSTH from former test_spike_train_stats.py

"""Focused suite: TestPSTH from former test_spike_train_stats.py."""

from __future__ import annotations

from tests.spike_train_stats_support import *  # noqa: F403

class TestPSTH:
    def test_shape(self):
        trials = [_poisson_train(100.0, 0.5, seed=i) for i in range(10)]
        rates, centers = psth(trials, bin_ms=10.0)
        assert rates.size > 0
        assert rates.size == centers.size

    def test_empty(self):
        rates, centers = psth([])
        assert rates.size == 0
