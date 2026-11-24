# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPlanckGrid from former test_zero_coverage_a.py

"""Focused suite: TestPlanckGrid from former test_zero_coverage_a.py."""

from __future__ import annotations


class TestPlanckGrid:
    """The Planck-scale grid research module."""

    def test_bekenstein(self) -> None:
        """The Bekenstein bound is positive."""
        from eschaton.computronium import PlanckGrid

        g = PlanckGrid(volume_cm3=1.0, mass_kg=1.0)
        assert g.bekenstein_bound() > 0

    def test_bremermann(self) -> None:
        """The Bremermann limit is positive."""
        from eschaton.computronium import PlanckGrid

        g = PlanckGrid()
        assert g.bremermann_limit() > 0

    def test_simulate_step(self) -> None:
        """A simulation step returns a string (the check is on the type only)."""
        from eschaton.computronium import PlanckGrid

        g = PlanckGrid()
        assert isinstance(g.simulate_step(), str)
