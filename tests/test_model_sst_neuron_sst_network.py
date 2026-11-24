# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSSTNetwork from former test_model_sst_neuron.py

"""Focused suite: TestSSTNetwork from former test_model_sst_neuron.py."""

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
from sc_neurocore.neurons.models.sst_neuron import (
    SSTNeuron,
)


class TestSSTNetwork:
    """The SSTNeuron in a network."""

    def test_population_size(self) -> None:
        """A population holds the requested number of neurons."""
        assert Population(SSTNeuron, n=8, label="sst").n == 8

    def test_population_drives_spikes(self) -> None:
        """Population drives spikes."""
        pop = Population(SSTNeuron, n=5, label="sst")
        drive = PoissonInput(n=5, rate_hz=500.0, weight=6.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=2.0, dt=0.001, backend="python")
        assert mon.count > 0
