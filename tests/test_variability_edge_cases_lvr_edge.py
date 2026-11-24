# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLvrEdge from former test_variability_edge_cases.py

"""Focused suite: TestLvrEdge from former test_variability_edge_cases.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.analysis.spike_stats.variability import (
    lvr,
)


class TestLvrEdge:
    """Edge cases of the revised local variation."""

    def test_empty(self) -> None:
        """The revised local variation of a silent train is NaN."""
        result = lvr(np.zeros(50, dtype=np.int8))
        assert np.isnan(result)

    def test_regular(self) -> None:
        """The revised local variation of a regular train is finite."""
        train = np.zeros(100, dtype=np.int8)
        train[::5] = 1
        result = lvr(train)
        assert np.isfinite(result)
