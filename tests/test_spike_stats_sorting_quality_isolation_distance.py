# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestIsolationDistance from former test_spike_stats_sorting_quality.py

"""Focused suite: TestIsolationDistance from former test_spike_stats_sorting_quality.py."""

from __future__ import annotations

from tests.spike_stats_sorting_quality_support import *  # noqa: F403


class TestIsolationDistance:
    def test_typical(self) -> None:
        rng = _rng()
        cluster = rng.normal(0, 1, (20, 3))
        noise = rng.normal(5, 1, (30, 3))
        result = isolation_distance(cluster, noise)
        assert np.isfinite(result)

    def test_too_small_cluster(self) -> None:
        result = isolation_distance(np.array([[1, 2]]), np.array([[3, 4], [5, 6]]))
        assert np.isnan(result)

    def test_fewer_noise_than_cluster(self) -> None:
        cluster = _rng().normal(0, 1, (10, 2))
        noise = _rng().normal(3, 1, (4, 2))
        assert np.isnan(isolation_distance(cluster, noise))

    def test_single_feature(self) -> None:
        cluster = _rng().normal(0, 1, (10, 1))
        noise = _rng().normal(3, 1, (20, 1))
        result = isolation_distance(cluster, noise)
        assert np.isfinite(result)

    def test_python_backend(self) -> None:
        cluster = _rng().normal(0, 1, (12, 2))
        noise = _rng().normal(4, 1, (40, 2))
        result = isolation_distance(cluster, noise, backend="python")
        assert np.isfinite(result) and result > 0.0
