# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestISI from former test_spike_train_stats.py

"""Focused suite: TestISI from former test_spike_train_stats.py."""

from __future__ import annotations

from tests.spike_train_stats_support import *  # noqa: F403


class TestISI:
    def test_regular(self):
        train = np.zeros(100, dtype=np.uint8)
        train[10::20] = 1
        intervals = isi(train, dt=0.001)
        np.testing.assert_allclose(intervals, 0.02, atol=1e-10)

    def test_single_spike(self):
        train = np.zeros(50, dtype=np.uint8)
        train[25] = 1
        assert isi(train).size == 0
