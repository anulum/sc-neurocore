# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestVoxelGrid from former test_research_modules.py

"""Focused suite: TestVoxelGrid from former test_research_modules.py."""

from __future__ import annotations

import sys
from pathlib import Path as _Path

# Ensure same-dir support module is importable under pytest importlib mode.
sys.path.insert(0, str(_Path(__file__).resolve().parent))

from research_modules_support import *  # noqa: F403

class TestVoxelGrid:
    def test_construction(self):
        vg = VoxelGrid(resolution=4)
        assert vg.data.shape == (4, 4, 4)
        assert np.all(vg.data == 0)

    def test_set_voxel(self):
        vg = VoxelGrid(resolution=4)
        vg.set_voxel(1, 2, 3, 0.8)
        assert vg.data[1, 2, 3] == 0.8

    def test_set_voxel_out_of_bounds(self):
        vg = VoxelGrid(resolution=4)
        vg.set_voxel(10, 0, 0, 1.0)  # should be no-op
        assert np.all(vg.data == 0)

    def test_get_as_bitstream(self):
        vg = VoxelGrid(resolution=2)
        vg.set_voxel(0, 0, 0, 1.0)
        bs = vg.get_as_bitstream(length=100)
        assert bs.shape == (2, 2, 2, 100)
        # Voxel at (0,0,0) should be all 1s (p=1.0)
        assert bs[0, 0, 0, :].mean() == 1.0
