# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPointCloud from former test_research_modules.py

"""Focused suite: TestPointCloud from former test_research_modules.py."""

from __future__ import annotations

from sc_neurocore.spatial.representations import PointCloud
import numpy as np


class TestPointCloud:
    """The point-cloud representation."""

    def test_normalize(self) -> None:
        """Normalisation brings the points and intensities into [0, 1]."""
        pts = np.array([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]])
        ints = np.array([0.5, 1.5])
        pc = PointCloud(points=pts, intensities=ints)
        pc.normalize()
        assert pc.points.min() >= 0.0
        assert pc.points.max() <= 1.0
        assert np.all(pc.intensities <= 1.0)
