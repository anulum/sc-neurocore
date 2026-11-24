# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBoothRinzelAnalysis from former test_model_booth_rinzel.py

"""Focused suite: TestBoothRinzelAnalysis from former test_model_booth_rinzel.py."""

from __future__ import annotations

from typing import Any
import numpy as np
from sc_neurocore.analysis.spike_stats.basic import (
    firing_rate,
    isi,
    spike_count,
)
from sc_neurocore.neurons.models.booth_rinzel import (
    BoothRinzelNeuron,
)


class TestBoothRinzelAnalysis:
    """Spike analysis of the BoothRinzelNeuron."""

    def _get_binary_train(self) -> np.ndarray[Any, Any]:
        n = BoothRinzelNeuron()
        train = np.zeros(50_000, dtype=np.int8)
        for t in range(50_000):
            train[t] = n.step(10.0)
        return train

    def test_firing_rate(self) -> None:
        """The firing rate of a driven spike train is computed."""
        train = self._get_binary_train()
        rate = firing_rate(train, dt=0.000025)  # dt=0.025ms (4 sub-steps)
        assert rate > 0

    def test_spike_count(self) -> None:
        """The spikes of a driven train are counted."""
        train = self._get_binary_train()
        assert spike_count(train) > 100

    def test_isi(self) -> None:
        """The inter-spike intervals of a driven train are finite."""
        train = self._get_binary_train()
        intervals = isi(train, dt=0.000025)
        if intervals.size > 0:
            assert np.all(np.isfinite(intervals))
            assert np.all(intervals > 0)
