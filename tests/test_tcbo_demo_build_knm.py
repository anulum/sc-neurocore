# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBuildKnm from former test_tcbo_demo.py

"""Focused suite: TestBuildKnm from former test_tcbo_demo.py."""

from __future__ import annotations

import numpy as np
import unittest
from sc_neurocore.scpn.params import (
    build_knm_matrix as _build_knm,
)


class TestBuildKnm(unittest.TestCase):
    """The coupling matrix of the TCBO demo."""

    def test_symmetric(self) -> None:
        """The 16-node coupling matrix is symmetric."""
        K = _build_knm(16)
        self.assertTrue(np.allclose(K, K.T))

    def test_zero_diagonal(self) -> None:
        """The coupling matrix has a zero diagonal."""
        K = _build_knm(16)
        self.assertTrue(np.allclose(np.diag(K), 0))

    def test_non_negative(self) -> None:
        """The coupling matrix is non-negative."""
        K = _build_knm(16)
        self.assertTrue(np.all(K >= 0))

    def test_small_N(self) -> None:
        """A four-node coupling matrix is 4x4 and symmetric."""
        K = _build_knm(4)
        self.assertEqual(K.shape, (4, 4))
        self.assertTrue(np.allclose(K, K.T))
