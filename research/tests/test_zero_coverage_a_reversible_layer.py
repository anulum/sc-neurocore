# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestReversibleLayer from former test_zero_coverage_a.py

"""Focused suite: TestReversibleLayer from former test_zero_coverage_a.py."""

from __future__ import annotations

import sys
from pathlib import Path as _Path

# Ensure same-dir support module is importable under pytest importlib mode.
sys.path.insert(0, str(_Path(__file__).resolve().parent))

from zero_coverage_a_support import *  # noqa: F403

class TestReversibleLayer:
    def test_toffoli(self):
        from post_silicon.reversible import ReversibleLayer

        g = ReversibleLayer()
        a = np.array([1, 0, 1, 0], dtype=np.uint8)
        b = np.array([0, 1, 1, 0], dtype=np.uint8)
        c = np.array([0, 0, 0, 0], dtype=np.uint8)
        a2, b2, c2 = g.toffoli_gate(a, b, c)
        assert c2.shape == c.shape
        a3, b3, c3 = g.reverse_toffoli(a2, b2, c2)
        np.testing.assert_array_equal(a3, a)

    def test_forward(self):
        from post_silicon.reversible import ReversibleLayer

        g = ReversibleLayer()
        a = np.array([1, 0, 1], dtype=np.uint8)
        b = np.array([0, 1, 1], dtype=np.uint8)
        out_a, out_b = g.forward(a, b)
        assert out_a.shape == a.shape
