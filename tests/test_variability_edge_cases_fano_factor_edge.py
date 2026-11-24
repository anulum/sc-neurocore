# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFanoFactorEdge from former test_variability_edge_cases.py

"""Focused suite: TestFanoFactorEdge from former test_variability_edge_cases.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.analysis.spike_stats.variability import (
    fano_factor,
)


class TestFanoFactorEdge:
    """Edge cases of the Fano factor."""

    def test_empty(self) -> None:
        """The Fano factor of a silent train is computed (the check only excludes infinity)."""
        result = fano_factor(np.zeros(50, dtype=np.int8))
        assert np.isfinite(result) or np.isnan(result)

    def test_all_spikes(self) -> None:
        """The Fano factor of an always-firing train is computed (the check only excludes infinity)."""
        result = fano_factor(np.ones(50, dtype=np.int8))
        assert np.isfinite(result) or np.isnan(result)
