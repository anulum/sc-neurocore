# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestVanRossumDistance from former test_spike_train_stats.py

"""Focused suite: TestVanRossumDistance from former test_spike_train_stats.py."""

from __future__ import annotations

from tests.spike_train_stats_support import *  # noqa: F403


class TestVanRossumDistance:
    def test_identical_zero(self):
        train = _poisson_train(100.0, 0.5)
        d = van_rossum_distance(train, train)
        assert d < 1e-6

    def test_different_positive(self):
        a = _poisson_train(100.0, 0.5, seed=1)
        b = _poisson_train(100.0, 0.5, seed=2)
        d = van_rossum_distance(a, b)
        assert d > 0
