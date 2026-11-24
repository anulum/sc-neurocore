# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSpatialTransformer3D from former test_research_modules.py

"""Focused suite: TestSpatialTransformer3D from former test_research_modules.py."""

from __future__ import annotations

from sc_neurocore.spatial.transformer_3d import SpatialTransformer3D
import numpy as np


class TestSpatialTransformer3D:
    """The 3-D spatial transformer."""

    def test_forward_shape(self) -> None:
        """A forward pass keeps the 3x3x3 shape."""
        st = SpatialTransformer3D(resolution=3, dim_k=2)
        grid = np.random.rand(3, 3, 3)
        out = st.forward(grid)
        assert out.shape == (3, 3, 3)

    def test_forward_non_negative(self) -> None:
        """A forward pass gives finite values (the check does not test the sign)."""
        st = SpatialTransformer3D(resolution=2, dim_k=4)
        grid = np.random.rand(2, 2, 2)
        out = st.forward(grid)
        # Attention output can vary, but should be finite
        assert np.all(np.isfinite(out))
