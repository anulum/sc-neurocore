# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFanoFactor from former test_spike_train_stats.py

"""Focused suite: TestFanoFactor from former test_spike_train_stats.py."""

from __future__ import annotations

from tests.spike_train_stats_support import *  # noqa: F403

class TestFanoFactor:
    def test_poisson_near_one(self):
        train = _poisson_train(100.0, 5.0)
        ff = fano_factor(train, window_ms=100.0)
        assert 0.5 < ff < 2.0

    def test_regular_below_one(self):
        train = np.zeros(5000, dtype=np.uint8)
        train[10::20] = 1
        ff = fano_factor(train, window_ms=100.0)
        assert ff < 0.5
