# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPresenceRatio from former test_spike_stats_sorting_quality.py

"""Focused suite: TestPresenceRatio from former test_spike_stats_sorting_quality.py."""

from __future__ import annotations

from tests.spike_stats_sorting_quality_support import *  # noqa: F403

class TestPresenceRatio:
    def test_full_presence(self) -> None:
        train = np.zeros(1000, dtype=np.int8)
        train[::10] = 1
        result = presence_ratio(train)
        assert result > 0.5

    def test_no_spikes(self) -> None:
        result = presence_ratio(np.zeros(100, dtype=np.int8))
        assert result == 0.0
