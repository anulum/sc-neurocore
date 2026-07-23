# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPlanckGrid from former test_zero_coverage_a.py

"""Focused suite: TestPlanckGrid from former test_zero_coverage_a.py."""

from __future__ import annotations

import sys
from pathlib import Path as _Path

# Ensure same-dir support module is importable under pytest importlib mode.
sys.path.insert(0, str(_Path(__file__).resolve().parent))

from zero_coverage_a_support import *  # noqa: F403

class TestPlanckGrid:
    def test_bekenstein(self):
        from eschaton.computronium import PlanckGrid

        g = PlanckGrid(volume_cm3=1.0, mass_kg=1.0)
        assert g.bekenstein_bound() > 0

    def test_bremermann(self):
        from eschaton.computronium import PlanckGrid

        g = PlanckGrid()
        assert g.bremermann_limit() > 0

    def test_simulate_step(self):
        from eschaton.computronium import PlanckGrid

        g = PlanckGrid()
        assert isinstance(g.simulate_step(), str)
