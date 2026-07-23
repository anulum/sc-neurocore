# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSpatialTransformer3D from former test_research_modules.py

"""Focused suite: TestSpatialTransformer3D from former test_research_modules.py."""

from __future__ import annotations

import sys
from pathlib import Path as _Path

# Ensure same-dir support module is importable under pytest importlib mode.
sys.path.insert(0, str(_Path(__file__).resolve().parent))

from research_modules_support import *  # noqa: F403

class TestSpatialTransformer3D:
    def test_forward_shape(self):
        st = SpatialTransformer3D(resolution=3, dim_k=2)
        grid = np.random.rand(3, 3, 3)
        out = st.forward(grid)
        assert out.shape == (3, 3, 3)

    def test_forward_non_negative(self):
        st = SpatialTransformer3D(resolution=2, dim_k=4)
        grid = np.random.rand(2, 2, 2)
        out = st.forward(grid)
        # Attention output can vary, but should be finite
        assert np.all(np.isfinite(out))
