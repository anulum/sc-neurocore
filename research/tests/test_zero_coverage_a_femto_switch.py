# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFemtoSwitch from former test_zero_coverage_a.py

"""Focused suite: TestFemtoSwitch from former test_zero_coverage_a.py."""

from __future__ import annotations

import sys
from pathlib import Path as _Path

# Ensure same-dir support module is importable under pytest importlib mode.
sys.path.insert(0, str(_Path(__file__).resolve().parent))

from zero_coverage_a_support import *  # noqa: F403

class TestFemtoSwitch:
    def test_interact(self):
        from post_silicon.femto import FemtoSwitch

        f = FemtoSwitch()
        a = np.array([1, 0, 1, 0], dtype=np.uint8)
        b = np.array([0, 1, 1, 0], dtype=np.uint8)
        result = f.interact(a, b)
        assert isinstance(result, np.ndarray)

    def test_bit_to_quark(self):
        from post_silicon.femto import FemtoSwitch

        f = FemtoSwitch()
        bs = np.array([1, 0, 1, 1], dtype=np.uint8)
        q = f.bit_to_quark(bs)
        assert isinstance(q, np.ndarray)
