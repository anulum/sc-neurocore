# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLRatio from former test_spike_stats_sorting_quality.py

"""Focused suite: TestLRatio from former test_spike_stats_sorting_quality.py."""

from __future__ import annotations

from tests.spike_stats_sorting_quality_support import *  # noqa: F403


class TestLRatio:
    def test_typical(self) -> None:
        rng = _rng()
        cluster = rng.normal(0, 1, (15, 2))
        noise = rng.normal(3, 1, (25, 2))
        result = l_ratio(cluster, noise)
        assert np.isfinite(result)

    def test_small_cluster(self) -> None:
        result = l_ratio(np.array([[1, 2]]), np.array([[3, 4]]))
        assert np.isnan(result)

    def test_empty_noise(self) -> None:
        cluster = _rng().normal(0, 1, (10, 2))
        result = l_ratio(cluster, np.empty((0, 2)))
        assert np.isnan(result)

    def test_single_feature(self) -> None:
        cluster = _rng().normal(0, 1, (10, 1))
        noise = _rng().normal(3, 1, (20, 1))
        result = l_ratio(cluster, noise)
        assert np.isfinite(result)

    def test_python_backend(self) -> None:
        cluster = _rng().normal(0, 1, (10, 2))
        noise = _rng().normal(3, 1, (30, 2))
        result = l_ratio(cluster, noise, backend="python")
        assert 0.0 <= result <= 1.0
