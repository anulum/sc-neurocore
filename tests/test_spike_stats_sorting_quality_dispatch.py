# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDispatch from former test_spike_stats_sorting_quality.py

"""Focused suite: TestDispatch from former test_spike_stats_sorting_quality.py."""

from __future__ import annotations

from tests.spike_stats_sorting_quality_support import *  # noqa: F403


class TestDispatch:
    def test_python_matches_reference(self) -> None:
        cluster, noise = _cluster_noise(20, 30, 3)
        direct = _SQ._isolation_distance_python(
            np.ascontiguousarray(cluster), np.ascontiguousarray(noise)
        )
        routed = isolation_distance(cluster, noise, backend="python")
        npt.assert_allclose(routed, direct, atol=0)

    def test_l_ratio_python_matches_reference(self) -> None:
        cluster, noise = _cluster_noise(15, 25, 2)
        direct = _SQ._l_ratio_python(np.ascontiguousarray(cluster), np.ascontiguousarray(noise))
        routed = l_ratio(cluster, noise, backend="python")
        npt.assert_allclose(routed, direct, atol=0)

    def test_unknown_backend_isolation(self) -> None:
        cluster, noise = _cluster_noise(10, 20, 2)
        with pytest.raises(ValueError, match="not available"):
            isolation_distance(cluster, noise, backend="cuda")

    def test_unknown_backend_l_ratio(self) -> None:
        cluster, noise = _cluster_noise(10, 20, 2)
        with pytest.raises(ValueError, match="not available"):
            l_ratio(cluster, noise, backend="cuda")
