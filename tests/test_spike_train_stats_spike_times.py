# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSpikeTimes from former test_spike_train_stats.py

"""Focused suite: TestSpikeTimes from former test_spike_train_stats.py."""

from __future__ import annotations

from tests.spike_train_stats_support import *  # noqa: F403


class TestSpikeTimes:
    def test_basic(self):
        train = np.array([0, 1, 0, 0, 1, 0], dtype=np.uint8)
        t = spike_times(train, dt=0.001)
        assert len(t) == 2
        np.testing.assert_allclose(t, [0.001, 0.004])

    def test_empty(self):
        assert spike_times(np.zeros(10, dtype=np.uint8)).size == 0
