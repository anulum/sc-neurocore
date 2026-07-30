# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestEnergyLIF from former test_extended_neurons.py

"""Focused suite: TestEnergyLIF from former test_extended_neurons.py."""

from __future__ import annotations

from tests.extended_neurons_support import *  # noqa: F403
from sc_neurocore.neurons.models.sc_normalized_energy_lif import (
    SCNormalizedEnergyLIFNeuron as EnergyLIFNeuron,
)


class TestEnergyLIF:
    def test_fires(self):
        n = EnergyLIFNeuron()
        assert sum(n.step(30.0) for _ in range(200)) > 0

    def test_energy_depletes(self):
        n = EnergyLIFNeuron()
        for _ in range(100):
            n.step(30.0)
        assert n.epsilon < 1.0
