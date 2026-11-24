# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCazellesNetwork from former test_model_cazelles_map.py

"""Focused suite: TestCazellesNetwork from former test_model_cazelles_map.py."""

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
from sc_neurocore.neurons.models.cazelles_map import (
    CazellesMapNeuron,
)


class TestCazellesNetwork:
    """The CazellesMapNeuron in a network."""

    def test_population(self) -> None:
        """A population of the model is built."""
        pop = Population(CazellesMapNeuron, n=10, label="caz")
        assert pop.n == 10

    def test_network_spikes(self) -> None:
        """A driven network runs and is monitored."""
        pop = Population(CazellesMapNeuron, n=20, label="caz")
        mon = SpikeMonitor(pop)
        net = Network(pop, mon)
        net.run(duration=0.6, dt=0.001, backend="python")
        assert mon.count > 0

    def test_with_projection(self) -> None:
        """A projected network yields readable spike trains."""
        pop = Population(CazellesMapNeuron, n=10, label="caz")
        proj = Projection(pop, pop, weight=0.05, probability=0.3, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, proj, mon)
        assert isinstance(mon.spike_trains, dict)
