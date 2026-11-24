# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCv2Edge from former test_variability_edge_cases.py

"""Focused suite: TestCv2Edge from former test_variability_edge_cases.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.analysis.spike_stats.variability import (
    cv2,
)


class TestCv2Edge:
    """Edge cases of the CV2."""

    def test_empty(self) -> None:
        """The CV2 of a silent train is NaN."""
        result = cv2(np.zeros(50, dtype=np.int8))
        assert np.isnan(result)

    def test_single_spike(self) -> None:
        """The CV2 of a single spike is NaN."""
        train = np.zeros(50, dtype=np.int8)
        train[25] = 1
        result = cv2(train)
        assert np.isnan(result)

    def test_regular_train(self) -> None:
        """The CV2 of a regular train is finite."""
        train = np.zeros(100, dtype=np.int8)
        train[::10] = 1
        result = cv2(train)
        assert np.isfinite(result)
