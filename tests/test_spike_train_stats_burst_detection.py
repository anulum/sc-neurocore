# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBurstDetection from former test_spike_train_stats.py

"""Focused suite: TestBurstDetection from former test_spike_train_stats.py."""

from __future__ import annotations

from tests.spike_train_stats_support import *  # noqa: F403

class TestBurstDetection:
    def test_detects_burst(self):
        train = np.zeros(1000, dtype=np.uint8)
        train[100:106] = 1
        bursts = burst_detection(train, dt=0.001, max_isi_ms=2.0, min_spikes=3)
        assert len(bursts) >= 1
        assert bursts[0][2] >= 3

    def test_no_burst_in_regular(self):
        train = np.zeros(1000, dtype=np.uint8)
        train[::50] = 1
        bursts = burst_detection(train, dt=0.001, max_isi_ms=5.0, min_spikes=3)
        assert len(bursts) == 0
