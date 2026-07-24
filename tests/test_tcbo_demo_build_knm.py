# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBuildKnm from former test_tcbo_demo.py

"""Focused suite: TestBuildKnm from former test_tcbo_demo.py."""

from __future__ import annotations

from tests.tcbo_demo_support import *  # noqa: F403


class TestBuildKnm(unittest.TestCase):
    def test_symmetric(self):
        K = _build_knm(16)
        self.assertTrue(np.allclose(K, K.T))

    def test_zero_diagonal(self):
        K = _build_knm(16)
        self.assertTrue(np.allclose(np.diag(K), 0))

    def test_non_negative(self):
        K = _build_knm(16)
        self.assertTrue(np.all(K >= 0))

    def test_small_N(self):
        K = _build_knm(4)
        self.assertEqual(K.shape, (4, 4))
        self.assertTrue(np.allclose(K, K.T))
