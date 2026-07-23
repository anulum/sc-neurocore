# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCellularComputer from former test_zero_coverage_a.py

"""Focused suite: TestCellularComputer from former test_zero_coverage_a.py."""

from __future__ import annotations

import sys
from pathlib import Path as _Path

# Ensure same-dir support module is importable under pytest importlib mode.
sys.path.insert(0, str(_Path(__file__).resolve().parent))

from zero_coverage_a_support import *  # noqa: F403

class TestCellularComputer:
    def test_step(self):
        from post_silicon.synthetic_cell import CellularComputer

        c = CellularComputer(n_molecules_a=10, n_molecules_b=5, reaction_rate=0.1)
        result = c.step(inject_a=2, inject_b=1)
        assert isinstance(result, (int, np.integer))
