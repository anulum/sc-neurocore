# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPowerBudgetAnalyzer from former test_photonic_noc.py

"""Focused suite: TestPowerBudgetAnalyzer from former test_photonic_noc.py."""

from __future__ import annotations

from photonic_noc_support import *  # noqa: F403

class TestPowerBudgetAnalyzer:
    """Power budget analysis tests."""

    def test_analyze(self, simple_design: PhotonicCircuitDesign) -> None:
        pba = PowerBudgetAnalyzer()
        result = pba.analyze(simple_design)
        assert result["n_paths"] > 0
        assert "worst_margin_db" in result

    def test_all_paths_have_margin(self, simple_design: PhotonicCircuitDesign) -> None:
        result = PowerBudgetAnalyzer().analyze(simple_design)
        for path in result["paths"]:
            assert "margin_db" in path
            assert "passed" in path
