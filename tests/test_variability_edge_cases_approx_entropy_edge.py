# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestApproxEntropyEdge from former test_variability_edge_cases.py

"""Focused suite: TestApproxEntropyEdge from former test_variability_edge_cases.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.analysis.spike_stats.variability import (
    approximate_entropy,
)


class TestApproxEntropyEdge:
    """Edge cases of approximate entropy."""

    def test_short_train(self) -> None:
        """Approximate entropy of a three-sample train is NaN."""
        result = approximate_entropy(np.zeros(3, dtype=np.int8))
        assert np.isnan(result)

    def test_constant(self) -> None:
        """Approximate entropy of a silent train is computed (the check only excludes infinity)."""
        result = approximate_entropy(np.zeros(100, dtype=np.int8))
        assert np.isfinite(result) or np.isnan(result)

    def test_normal(self) -> None:
        """Approximate entropy of a regular train is computed (the check only excludes infinity)."""
        train = np.zeros(200, dtype=np.int8)
        train[::5] = 1
        result = approximate_entropy(train)
        assert np.isfinite(result) or np.isnan(result)
