# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestNNHitRate from former test_spike_stats_sorting_quality.py

"""Focused suite: TestNNHitRate from former test_spike_stats_sorting_quality.py."""

from __future__ import annotations

from tests.spike_stats_sorting_quality_support import *  # noqa: F403

class TestNNHitRate:
    def test_typical(self) -> None:
        rng = _rng()
        cluster = rng.normal(0, 0.5, (20, 3))
        noise = rng.normal(5, 0.5, (20, 3))
        result = nn_hit_rate(cluster, noise, k=4)
        assert 0 <= result <= 1

    def test_too_small(self) -> None:
        cluster = _rng().normal(0, 1, (3, 2))
        noise = _rng().normal(3, 1, (10, 2))
        result = nn_hit_rate(cluster, noise, k=4)
        assert np.isnan(result)
