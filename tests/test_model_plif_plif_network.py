# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPLIFNetwork from former test_model_plif.py

"""Focused suite: TestPLIFNetwork from former test_model_plif.py."""

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
from sc_neurocore.neurons.models.plif import (
    ParametricLIFNeuron,
)


class TestPLIFNetwork:
    """The ParametricLIFNeuron in a network."""

    def test_population(self) -> None:
        """A population of the model is built."""
        pop = Population(ParametricLIFNeuron, n=10, label="plif")
        assert pop.n == 10

    def test_network_spikes(self) -> None:
        """A driven network runs and is monitored."""
        pop = Population(ParametricLIFNeuron, n=20, label="plif")
        drive = PoissonInput(n=20, rate_hz=500.0, weight=1.5, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon.count > 0
