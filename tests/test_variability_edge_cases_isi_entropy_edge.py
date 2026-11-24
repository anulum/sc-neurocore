# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestIsiEntropyEdge from former test_variability_edge_cases.py

"""Focused suite: TestIsiEntropyEdge from former test_variability_edge_cases.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.analysis.spike_stats.variability import (
    isi_entropy,
)


class TestIsiEntropyEdge:
    """Edge cases of the ISI entropy."""

    def test_empty(self) -> None:
        """The ISI entropy of a silent train is NaN or zero."""
        result = isi_entropy(np.zeros(50, dtype=np.int8))
        assert np.isnan(result) or result == 0.0

    def test_single_isi(self) -> None:
        """The ISI entropy of a single interval is computed (the check only excludes infinity)."""
        train = np.zeros(50, dtype=np.int8)
        train[10] = 1
        train[20] = 1
        result = isi_entropy(train)
        assert np.isfinite(result) or np.isnan(result)
