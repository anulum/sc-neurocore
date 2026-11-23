# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPowerBudgetAnalyzer from former test_bridges_photonic_noc.py

"""Focused suite: TestPowerBudgetAnalyzer from former test_bridges_photonic_noc.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.bridges.photonic_noc import (
    PowerBudgetAnalyzer,
    SCToPhotonic,
)


class TestPowerBudgetAnalyzer:
    """Smoke test of the power budget analysis."""

    def test_analyze_returns_dict(self) -> None:
        """The result is a dict; its contents are not checked."""
        compiler = SCToPhotonic()
        adj = np.array([[0, 1], [0, 0]], dtype=float)
        design = compiler.compile(adj)
        analyzer = PowerBudgetAnalyzer()
        result = analyzer.analyze(design)
        assert isinstance(result, dict)
