# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSCC from former test_diagnostics.py

"""Focused suite: TestSCC from former test_diagnostics.py."""

from __future__ import annotations

from diagnostics_support import *  # noqa: F403

class TestSCC(unittest.TestCase):
    """SCC computation tests."""

    def test_identical_streams(self):
        a = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 0], dtype=np.uint8)
        scc = compute_scc(a, a)
        self.assertAlmostEqual(scc, 1.0, places=5)

    def test_anticorrelated_streams(self):
        a = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.uint8)
        b = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.uint8)
        scc = compute_scc(a, b)
        self.assertAlmostEqual(scc, -1.0, places=5)

    def test_independent_streams(self):
        rng = np.random.default_rng(42)
        a = rng.integers(0, 2, size=10000, dtype=np.uint8)
        b = rng.integers(0, 2, size=10000, dtype=np.uint8)
        scc = compute_scc(a, b)
        self.assertLess(abs(scc), 0.1)
