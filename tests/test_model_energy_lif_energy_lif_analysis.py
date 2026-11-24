# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestEnergyLIFAnalysis from former test_model_energy_lif.py

"""Focused suite: TestEnergyLIFAnalysis from former test_model_energy_lif.py."""

from __future__ import annotations

from typing import Any
import numpy as np
from sc_neurocore.analysis.spike_stats.basic import (
    firing_rate,
    spike_count,
)
from sc_neurocore.neurons.models.sc_normalized_energy_lif import (
    SCNormalizedEnergyLIFNeuron as EnergyLIFNeuron,
)


class TestEnergyLIFAnalysis:
    """Spike analysis of the EnergyLIFNeuron."""

    def _get_train(self) -> np.ndarray[Any, Any]:
        n = EnergyLIFNeuron()
        train = np.zeros(5000, dtype=np.int8)
        for t in range(5000):
            train[t] = n.step(40.0)
        return train

    def test_firing_rate(self) -> None:
        """The firing rate of a driven spike train is computed."""
        assert firing_rate(self._get_train(), dt=0.001) > 0

    def test_spike_count(self) -> None:
        """The spikes of a driven train are counted."""
        assert spike_count(self._get_train()) > 10
