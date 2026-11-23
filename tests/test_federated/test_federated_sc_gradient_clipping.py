# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestGradientClipping from former test_federated_sc.py

"""Focused suite: TestGradientClipping from former test_federated_sc.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.federated.federated_sc import (
    clip_gradients,
)


class TestGradientClipping:
    """Clipping gradients to a norm bound."""

    def test_clips_large_gradient(self) -> None:
        """A large gradient is clipped to the bound."""
        g = np.array([3.0, 4.0])  # norm=5
        clipped = clip_gradients(g, max_norm=1.0)
        assert np.linalg.norm(clipped) <= 1.0 + 1e-6

    def test_does_not_clip_small_gradient(self) -> None:
        """A gradient within the bound is unchanged."""
        g = np.array([0.1, 0.2])
        clipped = clip_gradients(g, max_norm=10.0)
        np.testing.assert_array_almost_equal(clipped, g)

    def test_preserves_direction(self) -> None:
        """Clipping keeps the gradient's direction."""
        g = np.array([6.0, 8.0])  # norm=10
        clipped = clip_gradients(g, max_norm=5.0)
        direction = g / np.linalg.norm(g)
        clipped_dir = clipped / np.linalg.norm(clipped)
        np.testing.assert_array_almost_equal(direction, clipped_dir)

    def test_zero_gradient(self) -> None:
        """A zero gradient stays zero."""
        g = np.array([0.0, 0.0])
        clipped = clip_gradients(g, max_norm=1.0)
        np.testing.assert_array_equal(clipped, g)
