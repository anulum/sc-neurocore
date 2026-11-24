# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPermutationEntropyEdge from former test_variability_edge_cases.py

"""Focused suite: TestPermutationEntropyEdge from former test_variability_edge_cases.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.analysis.spike_stats.variability import (
    permutation_entropy,
)


class TestPermutationEntropyEdge:
    """Edge cases of permutation entropy."""

    def test_short(self) -> None:
        """Permutation entropy of a three-sample train is computed (the check only excludes infinity)."""
        result = permutation_entropy(np.zeros(3, dtype=np.int8))
        assert np.isnan(result) or np.isfinite(result)

    def test_constant(self) -> None:
        """Permutation entropy of a silent train is computed (the check only excludes infinity)."""
        result = permutation_entropy(np.zeros(100, dtype=np.int8))
        assert np.isfinite(result) or np.isnan(result)
