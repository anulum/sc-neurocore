# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestAllanFactorEdge from former test_variability_edge_cases.py

"""Focused suite: TestAllanFactorEdge from former test_variability_edge_cases.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.analysis.spike_stats.variability import (
    allan_factor,
)


class TestAllanFactorEdge:
    """Edge cases of the Allan factor."""

    def test_short(self) -> None:
        """The Allan factor of a five-sample train returns an array (the check is on the type only)."""
        vals, windows = allan_factor(np.zeros(5, dtype=np.int8))
        assert isinstance(vals, np.ndarray)

    def test_empty_spikes(self) -> None:
        """The Allan factor of a silent train returns an array (the check is on the type only)."""
        vals, windows = allan_factor(np.zeros(200, dtype=np.int8))
        assert isinstance(vals, np.ndarray)

    def test_normal(self) -> None:
        """The Allan factor of a regular train has values."""
        train = np.zeros(500, dtype=np.int8)
        train[::10] = 1
        vals, windows = allan_factor(train)
        assert len(vals) > 0
