# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestVictorPurpuraDistance from former test_spike_train_stats.py

"""Focused suite: TestVictorPurpuraDistance from former test_spike_train_stats.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.analysis.spike_stats import (
    victor_purpura_distance,
)


class TestVictorPurpuraDistance:
    """The Victor-Purpura distance between spike-time sets."""

    def test_identical_zero(self) -> None:
        """A spike-time set is at distance zero from itself."""
        t = np.array([0.1, 0.2, 0.3])
        assert victor_purpura_distance(t, t) < 1e-10

    def test_empty(self) -> None:
        """An empty set is at a distance of two from two spikes."""
        assert victor_purpura_distance(np.array([]), np.array([0.1, 0.2])) == 2.0

    def test_different(self) -> None:
        """Spike times shifted by 50 ms are at a positive distance at a cost of 100 per s."""
        a = np.array([0.1, 0.3, 0.5])
        b = np.array([0.15, 0.35, 0.55])
        d = victor_purpura_distance(a, b, cost_per_s=100.0)
        assert d > 0
