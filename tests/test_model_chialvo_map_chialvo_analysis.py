# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestChialvoAnalysis from former test_model_chialvo_map.py

"""Focused suite: TestChialvoAnalysis from former test_model_chialvo_map.py."""

from __future__ import annotations

from typing import Any
import numpy as np
from sc_neurocore.analysis.spike_stats.basic import (
    firing_rate,
    isi,
    spike_count,
)
from sc_neurocore.neurons.models.chialvo_map import (
    ChialvoMapNeuron,
)


class TestChialvoAnalysis:
    """Spike analysis of the ChialvoMapNeuron."""

    def _get_train(self) -> np.ndarray[Any, Any]:
        n = ChialvoMapNeuron()
        train = np.zeros(5000, dtype=np.int8)
        for t in range(5000):
            train[t] = n.step(0.0)
        return train

    def test_firing_rate(self) -> None:
        """The firing rate of a driven spike train is computed."""
        train = self._get_train()
        rate = firing_rate(train, dt=0.001)
        assert rate > 0

    def test_spike_count(self) -> None:
        """The spikes of a driven train are counted."""
        train = self._get_train()
        assert spike_count(train) > 0

    def test_isi(self) -> None:
        """The inter-spike intervals of a driven train are finite."""
        train = self._get_train()
        intervals = isi(train, dt=0.001)
        if intervals.size > 0:
            assert np.all(np.isfinite(intervals))
