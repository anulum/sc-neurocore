# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPlantR15Network from former test_model_plant_r15.py

"""Focused suite: TestPlantR15Network from former test_model_plant_r15.py."""

from __future__ import annotations

from sc_neurocore.network.monitor import (
    SpikeMonitor,
)
from sc_neurocore.network.network import (
    Network,
)
from sc_neurocore.network.population import (
    Population,
)
from sc_neurocore.network.stimulus import (
    PoissonInput,
)
from sc_neurocore.neurons.models.plant_r15 import (
    PlantR15Neuron,
)


class TestPlantR15Network:
    """The PlantR15Neuron in a network."""

    def test_population(self) -> None:
        """A population of the model is built."""
        pop = Population(PlantR15Neuron, n=5, label="r15")
        assert pop.n == 5

    def test_network_spikes(self) -> None:
        """With strong Poisson drive, R15 neurons should fire."""
        pop = Population(PlantR15Neuron, n=5, label="r15")
        drive = PoissonInput(n=5, rate_hz=500.0, weight=10.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=2.0, dt=0.001, backend="python")
        assert mon.count > 0
