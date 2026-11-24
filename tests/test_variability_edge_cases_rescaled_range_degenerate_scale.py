# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestRescaledRangeDegenerateScale from former test_variability_edge_cases.py

"""Focused suite: TestRescaledRangeDegenerateScale from former test_variability_edge_cases.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.analysis.spike_stats.variability import (
    rescaled_range,
)


class TestRescaledRangeDegenerateScale:
    """The rescaled range terminates when its first window step stalls.

    min_window == 1 makes the first 1.5x step stall at 1; the unit-step guard
    must force progress so the analysis terminates instead of looping forever.
    """

    def test_min_window_one_terminates(self) -> None:
        """A minimum window of one terminates with a finite rescaled range."""
        train = np.zeros(300, dtype=np.int8)
        train[::3] = 1
        result = rescaled_range(train, min_window=1)
        assert np.isfinite(result)
