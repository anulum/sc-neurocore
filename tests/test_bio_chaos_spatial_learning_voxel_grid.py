# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestVoxelGrid from former test_bio_chaos_spatial_learning.py

"""Focused suite: TestVoxelGrid from former test_bio_chaos_spatial_learning.py."""

from __future__ import annotations

from tests.bio_chaos_spatial_learning_support import *  # noqa: F403


class TestVoxelGrid:
    def test_init_zeros(self):
        vg = VoxelGrid(resolution=4)
        assert vg.data.shape == (4, 4, 4)
        assert np.all(vg.data == 0)

    def test_set_voxel(self):
        vg = VoxelGrid(resolution=4)
        vg.set_voxel(1, 2, 3, 0.9)
        assert vg.data[1, 2, 3] == 0.9

    def test_set_voxel_out_of_bounds(self):
        vg = VoxelGrid(resolution=4)
        vg.set_voxel(10, 10, 10, 1.0)
        assert np.all(vg.data == 0)

    def test_bitstream_shape(self):
        bs = VoxelGrid(resolution=2).get_as_bitstream(length=64)
        assert bs.shape == (2, 2, 2, 64)
        assert bs.dtype == np.uint8
