# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestAdaptiveThresholdIFAnalysis from former test_model_adaptive_threshold_if.py

"""Focused suite: TestAdaptiveThresholdIFAnalysis from former test_model_adaptive_threshold_if.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.analysis.spike_stats.basic import (
    firing_rate,
    isi,
    spike_count,
)
from sc_neurocore.neurons.models.adaptive_threshold_if import (
    AdaptiveThresholdIFNeuron,
)


class TestAdaptiveThresholdIFAnalysis:
    """Analysis toolkit works on spikes from this model."""

    def _get_binary_train(self) -> np.ndarray:
        n = AdaptiveThresholdIFNeuron()
        train = np.zeros(5000, dtype=np.int8)
        for t in range(5000):
            train[t] = n.step(80.0)
        return train

    def test_firing_rate(self) -> None:
        """Driven firing gives a positive firing rate."""
        train = self._get_binary_train()
        rate = firing_rate(train, dt=0.0001)  # dt=0.1ms (model dt)
        assert rate > 0

    def test_spike_count(self) -> None:
        """A driven spike train has spikes."""
        train = self._get_binary_train()
        assert spike_count(train) > 0

    def test_isi(self) -> None:
        """Inter-spike intervals are positive."""
        train = self._get_binary_train()
        intervals = isi(train, dt=0.0001)
        if intervals.size > 0:
            assert np.all(intervals > 0)
