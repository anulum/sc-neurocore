# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPowerBudgetAnalyzer from former test_photonic_noc.py

"""Focused suite: TestPowerBudgetAnalyzer from former test_photonic_noc.py."""

from __future__ import annotations

from sc_neurocore.bridges.photonic_noc import (
    PhotonicCircuitDesign,
    PowerBudgetAnalyzer,
)


class TestPowerBudgetAnalyzer:
    """Power budget analysis tests."""

    def test_analyze(self, simple_design: PhotonicCircuitDesign) -> None:
        """The power budget covers paths and a worst margin."""
        pba = PowerBudgetAnalyzer()
        result = pba.analyze(simple_design)
        assert result["n_paths"] > 0
        assert "worst_margin_db" in result

    def test_all_paths_have_margin(self, simple_design: PhotonicCircuitDesign) -> None:
        """Each path reports its margin and a verdict."""
        result = PowerBudgetAnalyzer().analyze(simple_design)
        for path in result["paths"]:
            assert "margin_db" in path
            assert "passed" in path
