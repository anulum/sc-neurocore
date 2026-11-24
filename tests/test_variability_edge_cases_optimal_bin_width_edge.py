# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestOptimalBinWidthEdge from former test_variability_edge_cases.py

"""Focused suite: TestOptimalBinWidthEdge from former test_variability_edge_cases.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.analysis.spike_stats.variability import (
    optimal_bin_width,
)


class TestOptimalBinWidthEdge:
    """Edge cases of the optimal bin width."""

    def test_empty(self) -> None:
        """The optimal bin width of a silent train is computed (the check only excludes infinity)."""
        result = optimal_bin_width(np.zeros(50, dtype=np.int8))
        assert np.isnan(result) or np.isfinite(result)

    def test_single_spike(self) -> None:
        """The optimal bin width of a single spike is computed (the check only excludes infinity)."""
        train = np.zeros(50, dtype=np.int8)
        train[25] = 1
        result = optimal_bin_width(train)
        assert np.isnan(result) or np.isfinite(result)

    def test_normal(self) -> None:
        """The optimal bin width of a regular train is finite."""
        train = np.zeros(200, dtype=np.int8)
        train[::10] = 1
        result = optimal_bin_width(train)
        assert np.isfinite(result)
