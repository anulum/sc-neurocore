# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestHolographicBoundary from former test_zero_coverage_a.py

"""Focused suite: TestHolographicBoundary from former test_zero_coverage_a.py."""

from __future__ import annotations

import numpy as np


class TestHolographicBoundary:
    """The holographic boundary research module."""

    def test_encode_reconstruct(self) -> None:
        """Encoding and reconstruction both return a value (the check is that they are not None)."""
        from eschaton.holographic import HolographicBoundary

        h = HolographicBoundary(grid_size=4)
        bulk = np.random.randn(4, 4, 4)
        encoded = h.encode_to_boundary(bulk)
        assert encoded is not None
        reconstructed = h.reconstruct_bulk()
        assert reconstructed is not None
