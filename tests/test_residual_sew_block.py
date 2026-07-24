# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSEWBlock from former test_residual.py

"""Focused suite: TestSEWBlock from former test_residual.py."""

from __future__ import annotations

from tests.residual_support import *  # noqa: F403


class TestSEWBlock:
    def test_forward(self) -> None:
        b = SEWBlock(n_features=8)
        out = b.forward(np.random.rand(8))
        assert out.shape == (8,) and out.max() <= 1.0

    def test_reset(self) -> None:
        """Reset should clear the SEW membrane state."""
        b = SEWBlock(n_features=4)
        b._v[:] = 0.5
        b.reset()
        assert np.allclose(b._v, 0)
