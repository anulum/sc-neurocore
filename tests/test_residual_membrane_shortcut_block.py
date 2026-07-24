# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestMembraneShortcutBlock from former test_residual.py

"""Focused suite: TestMembraneShortcutBlock from former test_residual.py."""

from __future__ import annotations

from tests.residual_support import *  # noqa: F403


class TestMembraneShortcutBlock:
    def test_forward(self) -> None:
        b = MembraneShortcutBlock(n_features=8)
        assert b.forward(np.random.rand(8)).shape == (8,)

    def test_reset(self) -> None:
        b = MembraneShortcutBlock(n_features=4)
        b.forward(np.ones(4))
        b.reset()
        assert np.allclose(b._v, 0)
