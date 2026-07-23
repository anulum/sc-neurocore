# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPlantR15Network from former test_model_plant_r15.py

"""Focused suite: TestPlantR15Network from former test_model_plant_r15.py."""

from __future__ import annotations

from tests.model_plant_r15_support import *  # noqa: F403

class TestPlantR15Network:
    def test_population(self):
        pop = Population(PlantR15Neuron, n=5, label="r15")
        assert pop.n == 5

    def test_network_spikes(self):
        """With strong Poisson drive, R15 neurons should fire."""
        pop = Population(PlantR15Neuron, n=5, label="r15")
        drive = PoissonInput(n=5, rate_hz=500.0, weight=10.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=2.0, dt=0.001, backend="python")
        assert mon.count > 0
