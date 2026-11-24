# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSpiNNaker2Pipeline from former test_model_spinnaker2.py

"""Focused suite: TestSpiNNaker2Pipeline from former test_model_spinnaker2.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.analysis.spike_stats.basic import (
    spike_count,
)
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
from sc_neurocore.neurons.models.spinnaker2 import (
    SpiNNaker2Neuron,
)


class TestSpiNNaker2Pipeline:
    """The SpiNNaker2Neuron in the network pipeline."""

    def test_population_creates(self) -> None:
        """A population of the model is built."""
        assert Population(SpiNNaker2Neuron, n=10, label="sn2").n == 10

    def test_network_incompatible(self) -> None:
        """Running a network of SpiNNaker2Neuron fails, a known limitation.

        SpiNNaker2 uses the integer >> operator, and Population.step_all passes
        float(currents[i]), which fails on >>. Integer neuromorphic models need an
        int-cast adapter for Network.
        """
        import pytest

        pop = Population(SpiNNaker2Neuron, n=5, label="sn2")
        drive = PoissonInput(n=5, rate_hz=500.0, weight=500.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        with pytest.raises(TypeError):
            net.run(duration=0.1, dt=0.001, backend="python")

    def test_analysis(self) -> None:
        """The analysis reports spikes and a rate for a driven run."""
        n = SpiNNaker2Neuron()
        train = np.array([float(n.step(500)) for _ in range(5000)])
        assert spike_count(train) >= 10

    def test_deterministic(self) -> None:
        """The same inputs give the same trace."""
        traces = []
        for _ in range(2):
            n = SpiNNaker2Neuron()
            trace = [(n.step(500), n.v) for _ in range(200)]
            traces.append(trace)
        assert traces[0] == traces[1]
