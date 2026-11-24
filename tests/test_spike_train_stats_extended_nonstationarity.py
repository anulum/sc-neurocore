# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestNonstationarity from former test_spike_train_stats_extended.py

"""Focused suite: TestNonstationarity from former test_spike_train_stats_extended.py."""

from __future__ import annotations

from typing import Any

from sc_neurocore.analysis.spike_stats import change_point_detection
from sc_neurocore.analysis.spike_stats import cubic_higher_order
import numpy as np
from tests.spike_train_stats_extended_support import poisson_train as poisson_train


class TestNonstationarity:
    """Change-point detection and the cubic higher-order statistic."""

    def test_change_point_detection(self) -> None:
        """A rate step from 1 to 10 percent per step yields at least one change point."""
        t = np.zeros(2000)
        rng = np.random.default_rng(11)
        t[:1000] = (rng.random(1000) < 0.01).astype(np.float64)
        t[1000:] = (rng.random(1000) < 0.1).astype(np.float64)
        cps = change_point_detection(t, bin_size=50, threshold=3.0)
        assert isinstance(cps, list)
        assert len(cps) > 0

    def test_cubic_higher_order(self, poisson_train: np.ndarray[Any, Any]) -> None:
        """The cubic higher-order statistic with a maximum lag of 10 is 10x10."""
        c3 = cubic_higher_order(poisson_train, max_lag=10)
        assert c3.shape == (10, 10)
