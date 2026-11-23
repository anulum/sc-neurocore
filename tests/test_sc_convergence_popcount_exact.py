# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPopcountExact from former test_sc_convergence.py

"""Focused suite: TestPopcountExact from former test_sc_convergence.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore import (
    bitstream_to_probability,
)


class TestPopcountExact:
    """Popcount should return the exact number of 1-bits."""

    def test_known_popcount(self) -> None:
        """A known bitstream gives its exact popcount and probability."""
        bits = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
        assert np.sum(bits) == 4
        assert bitstream_to_probability(bits) == 0.5

    def test_all_ones(self) -> None:
        """An all-ones bitstream has probability one."""
        bits = np.ones(1000, dtype=np.uint8)
        assert bitstream_to_probability(bits) == 1.0

    def test_all_zeros(self) -> None:
        """An all-zeros bitstream has probability zero."""
        bits = np.zeros(1000, dtype=np.uint8)
        assert bitstream_to_probability(bits) == 0.0
