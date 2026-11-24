# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLocalVariationEdge from former test_variability_edge_cases.py

"""Focused suite: TestLocalVariationEdge from former test_variability_edge_cases.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.analysis.spike_stats.variability import (
    local_variation,
)


class TestLocalVariationEdge:
    """Edge cases of the local variation."""

    def test_empty(self) -> None:
        """The local variation of a silent train is NaN."""
        result = local_variation(np.zeros(50, dtype=np.int8))
        assert np.isnan(result)

    def test_regular(self) -> None:
        """The local variation of a regular train is finite."""
        train = np.zeros(100, dtype=np.int8)
        train[::5] = 1
        result = local_variation(train)
        assert np.isfinite(result)
