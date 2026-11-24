# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBrainScaleSNetwork from former test_model_brainscales_adex.py

"""Focused suite: TestBrainScaleSNetwork from former test_model_brainscales_adex.py."""

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
from sc_neurocore.neurons.models.brainscales_adex import (
    BrainScaleSAdExNeuron,
)


class TestBrainScaleSNetwork:
    """The BrainScaleSAdExNeuron in a network."""

    def test_population(self) -> None:
        """A population of the model is built."""
        pop = Population(BrainScaleSAdExNeuron, n=10, label="bs")
        assert pop.n == 10
        assert pop.model_name == "BrainScaleSAdExNeuron"

    def test_network_spikes(self) -> None:
        """A driven network runs and is monitored."""
        pop = Population(BrainScaleSAdExNeuron, n=20, label="bs")
        drive = PoissonInput(n=20, rate_hz=500.0, weight=40.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=0.5, dt=0.001, backend="python")
        assert mon.count > 0, "network zero spikes"

    def test_with_projection(self) -> None:
        """A projected network yields readable spike trains."""
        pop = Population(BrainScaleSAdExNeuron, n=10, label="bs")
        proj = Projection(pop, pop, weight=2.0, probability=0.3, seed=42)
        drive = PoissonInput(n=10, rate_hz=500.0, weight=40.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, proj, drive, mon)
        net.run(duration=0.5, dt=0.001, backend="python")
        assert isinstance(mon.spike_trains, dict)
