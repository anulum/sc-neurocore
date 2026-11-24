# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestEnergyLIFNetwork from former test_model_energy_lif.py

"""Focused suite: TestEnergyLIFNetwork from former test_model_energy_lif.py."""

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
from sc_neurocore.network.projection import (
    Projection,
)
from sc_neurocore.network.stimulus import (
    PoissonInput,
)
from sc_neurocore.neurons.models.sc_normalized_energy_lif import (
    SCNormalizedEnergyLIFNeuron as EnergyLIFNeuron,
)


class TestEnergyLIFNetwork:
    """The EnergyLIFNeuron in a network."""

    def test_population(self) -> None:
        """A population of the model is built."""
        assert Population(EnergyLIFNeuron, n=10, label="elif").n == 10

    def test_network_spikes(self) -> None:
        """A driven network runs and is monitored."""
        pop = Population(EnergyLIFNeuron, n=20, label="elif")
        drive = PoissonInput(n=20, rate_hz=500.0, weight=30.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=0.5, dt=0.001, backend="python")
        assert mon.count > 0

    def test_with_projection(self) -> None:
        """A projected network yields readable spike trains."""
        pop = Population(EnergyLIFNeuron, n=10, label="elif")
        proj = Projection(pop, pop, weight=5.0, probability=0.3, seed=42)
        drive = PoissonInput(n=10, rate_hz=500.0, weight=30.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, proj, drive, mon)
        net.run(duration=0.3, dt=0.001, backend="python")
        assert isinstance(mon.spike_trains, dict)
