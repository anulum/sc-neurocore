# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSpikeCount from former test_spike_train_stats.py

"""Focused suite: TestSpikeCount from former test_spike_train_stats.py."""

from __future__ import annotations

from tests.spike_train_stats_support import *  # noqa: F403


class TestSpikeCount:
    def test_count(self):
        train = np.array([1, 0, 1, 1, 0, 0, 1], dtype=np.uint8)
        assert spike_count(train) == 4
