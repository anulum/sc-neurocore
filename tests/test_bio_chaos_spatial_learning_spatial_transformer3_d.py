# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSpatialTransformer3D from former test_bio_chaos_spatial_learning.py

"""Focused suite: TestSpatialTransformer3D from former test_bio_chaos_spatial_learning.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.spatial.transformer_3d import (
    SpatialTransformer3D,
)


class TestSpatialTransformer3D:
    """Transforming 3-D voxel grids."""

    def test_output_shape(self) -> None:
        """The transform keeps the grid shape."""
        grid = np.random.rand(3, 3, 3)
        out = SpatialTransformer3D(resolution=3, dim_k=4).forward(grid)
        assert out.shape == (3, 3, 3)

    def test_output_differs(self) -> None:
        """The transform changes the grid."""
        grid = np.random.rand(3, 3, 3)
        out = SpatialTransformer3D(resolution=3, dim_k=4).forward(grid)
        assert not np.array_equal(grid, out)
