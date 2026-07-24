# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPowerBudgetAnalyzer from former test_bridges_photonic_noc.py

"""Focused suite: TestPowerBudgetAnalyzer from former test_bridges_photonic_noc.py."""

from __future__ import annotations

from tests.bridges_photonic_noc_support import *  # noqa: F403


class TestPowerBudgetAnalyzer:
    def test_analyze_returns_dict(self):
        compiler = SCToPhotonic()
        adj = np.array([[0, 1], [0, 0]], dtype=float)
        design = compiler.compile(adj)
        analyzer = PowerBudgetAnalyzer()
        result = analyzer.analyze(design)
        assert isinstance(result, dict)
