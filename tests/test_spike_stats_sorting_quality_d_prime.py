# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDPrime from former test_spike_stats_sorting_quality.py

"""Focused suite: TestDPrime from former test_spike_stats_sorting_quality.py."""

from __future__ import annotations

from tests.spike_stats_sorting_quality_support import *  # noqa: F403


class TestDPrime:
    def test_typical(self) -> None:
        rng = _rng()
        a = rng.normal(0, 1, (20, 3))
        b = rng.normal(3, 1, (20, 3))
        result = d_prime(a, b)
        assert result > 0

    def test_identical_clusters(self) -> None:
        data = _rng().normal(0, 1, (10, 2))
        result = d_prime(data, data.copy())
        assert result == 0.0

    def test_zero_variance(self) -> None:
        a = np.ones((5, 2))
        b = np.ones((5, 2)) * 2
        result = d_prime(a, b)
        assert result == 0.0 or np.isfinite(result)
