# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestIsiViolationRate from former test_spike_stats_sorting_quality.py

"""Focused suite: TestIsiViolationRate from former test_spike_stats_sorting_quality.py."""

from __future__ import annotations

from tests.spike_stats_sorting_quality_support import *  # noqa: F403

class TestIsiViolationRate:
    def test_no_violations(self) -> None:
        train = np.zeros(1000, dtype=np.int8)
        train[::100] = 1  # 10 Hz, ISI = 100 ms >> 1.5 ms
        result = isi_violation_rate(train)
        assert result == 0.0

    def test_empty(self) -> None:
        result = isi_violation_rate(np.zeros(100, dtype=np.int8))
        assert result == 0.0

    def test_all_violations(self) -> None:
        train = np.ones(10, dtype=np.int8)  # ISI = 1 ms < 1.5 ms
        result = isi_violation_rate(train)
        assert result > 0
