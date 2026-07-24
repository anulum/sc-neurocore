# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPointCloud from former test_bio_chaos_spatial_learning.py

"""Focused suite: TestPointCloud from former test_bio_chaos_spatial_learning.py."""

from __future__ import annotations

from tests.bio_chaos_spatial_learning_support import *  # noqa: F403


class TestPointCloud:
    def test_normalize(self):
        pc = PointCloud(
            points=np.array([[0.0, 10.0, 20.0], [5.0, 15.0, 25.0]]),
            intensities=np.array([0.5, 1.5]),
        )
        pc.normalize()
        assert np.min(pc.points) >= 0.0
        assert np.max(pc.points) <= 1.0 + 1e-9
        assert np.all(pc.intensities <= 1.0)
