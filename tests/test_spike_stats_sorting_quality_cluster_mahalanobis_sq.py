# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestClusterMahalanobisSq from former test_spike_stats_sorting_quality.py

"""Focused suite: TestClusterMahalanobisSq from former test_spike_stats_sorting_quality.py."""

from __future__ import annotations

from tests.spike_stats_sorting_quality_support import *  # noqa: F403

class TestClusterMahalanobisSq:
    def test_matches_closed_form_inverse(self) -> None:
        # The Cholesky-solve kernel must equal diffᵀ Σ⁻¹ diff with the dense
        # inverse, without ever forming Σ⁻¹ in the kernel itself.
        cluster = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0]])
        point = np.array([[5.0, 3.0]])
        mah = _SQ._cluster_mahalanobis_sq(cluster, point)
        cov = np.cov(cluster.T) + 1e-8 * np.eye(2)
        diff = point[0] - cluster.mean(axis=0)
        ref = float(diff @ np.linalg.inv(cov) @ diff)
        npt.assert_allclose(mah[0], ref, atol=1e-9)

    def test_centre_is_zero(self) -> None:
        cluster = np.array([[1.0, 4.0], [3.0, 4.0], [1.0, 8.0], [3.0, 8.0]])
        centre = cluster.mean(axis=0, keepdims=True)
        mah = _SQ._cluster_mahalanobis_sq(cluster, centre)
        assert abs(mah[0]) < 1e-9

    def test_single_feature(self) -> None:
        cluster = np.array([[1.0], [3.0], [5.0], [7.0]])
        noise = np.array([[10.0], [0.0]])
        mah = _SQ._cluster_mahalanobis_sq(cluster, noise)
        assert mah.shape == (2,)
        assert np.all(mah >= 0.0)
