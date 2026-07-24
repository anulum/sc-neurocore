# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFiringRate from former test_spike_train_stats.py

"""Focused suite: TestFiringRate from former test_spike_train_stats.py."""

from __future__ import annotations

from tests.spike_train_stats_support import *  # noqa: F403


class TestFiringRate:
    def test_known_rate(self):
        train = _poisson_train(100.0, 1.0)
        rate = firing_rate(train, dt=0.001)
        assert 70 < rate < 140

    def test_empty(self):
        assert firing_rate(np.zeros(100, dtype=np.uint8)) == 0.0
