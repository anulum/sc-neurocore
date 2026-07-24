# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSpikeTriggeredAverage from former test_spike_train_stats.py

"""Focused suite: TestSpikeTriggeredAverage from former test_spike_train_stats.py."""

from __future__ import annotations

from tests.spike_train_stats_support import *  # noqa: F403


class TestSpikeTriggeredAverage:
    def test_shape(self):
        stim = np.sin(np.linspace(0, 10 * np.pi, 1000))
        train = np.zeros(1000, dtype=np.uint8)
        train[100::100] = 1
        sta = spike_triggered_average(stim, train, window_steps=50)
        assert sta.shape == (50,)

    def test_no_spikes(self):
        sta = spike_triggered_average(np.ones(100), np.zeros(100, dtype=np.uint8))
        np.testing.assert_allclose(sta, 0.0)
