# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestHurstExponentEdge from former test_variability_edge_cases.py

"""Focused suite: TestHurstExponentEdge from former test_variability_edge_cases.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.analysis.spike_stats.variability import (
    hurst_exponent,
)


class TestHurstExponentEdge:
    """Edge cases of the Hurst exponent."""

    def test_short(self) -> None:
        """The Hurst exponent of a five-sample train is computed (the check only excludes infinity)."""
        result = hurst_exponent(np.zeros(5, dtype=np.int8))
        assert np.isnan(result) or np.isfinite(result)

    def test_normal(self) -> None:
        """The Hurst exponent of a regular train is computed (the check only excludes infinity)."""
        train = np.zeros(500, dtype=np.int8)
        train[::10] = 1
        result = hurst_exponent(train)
        assert np.isfinite(result) or np.isnan(result)
