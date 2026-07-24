# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestEnergyLIFAnalysis from former test_model_energy_lif.py

"""Focused suite: TestEnergyLIFAnalysis from former test_model_energy_lif.py."""

from __future__ import annotations

from tests.model_energy_lif_support import *  # noqa: F403


class TestEnergyLIFAnalysis:
    def _get_train(self):
        n = EnergyLIFNeuron()
        train = np.zeros(5000, dtype=np.int8)
        for t in range(5000):
            train[t] = n.step(40.0)
        return train

    def test_firing_rate(self):
        assert firing_rate(self._get_train(), dt=0.001) > 0

    def test_spike_count(self):
        assert spike_count(self._get_train()) > 10
