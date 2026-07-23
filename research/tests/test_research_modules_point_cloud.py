# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPointCloud from former test_research_modules.py

"""Focused suite: TestPointCloud from former test_research_modules.py."""

from __future__ import annotations

import sys
from pathlib import Path as _Path

# Ensure same-dir support module is importable under pytest importlib mode.
sys.path.insert(0, str(_Path(__file__).resolve().parent))

from research_modules_support import *  # noqa: F403

class TestPointCloud:
    def test_normalize(self):
        pts = np.array([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]])
        ints = np.array([0.5, 1.5])
        pc = PointCloud(points=pts, intensities=ints)
        pc.normalize()
        assert pc.points.min() >= 0.0
        assert pc.points.max() <= 1.0
        assert np.all(pc.intensities <= 1.0)
