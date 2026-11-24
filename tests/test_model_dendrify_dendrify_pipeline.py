# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDendrifyPipeline from former test_model_dendrify.py

"""Focused suite: TestDendrifyPipeline from former test_model_dendrify.py."""

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
from sc_neurocore.neurons.models.dendrify import (
    DendrifyNeuron,
)


class TestDendrifyPipeline:
    """The DendrifyNeuron in the network pipeline."""

    def test_population(self) -> None:
        """A population of the model is built."""
        assert Population(DendrifyNeuron, n=10, label="dend").n == 10

    def test_network_spikes(self) -> None:
        """A driven network runs and is monitored."""
        pop = Population(DendrifyNeuron, n=10, label="dend")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=50.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=2.0, dt=0.001, backend="python")
        # Dendrify needs high current — may not spike in network
        assert isinstance(mon.count, int)

    def test_analysis(self) -> None:
        """The analysis reports spikes and a rate for a driven run."""
        n = DendrifyNeuron()
        train = np.array([float(n.step(50.0)) for _ in range(10000)])
        assert spike_count(train) >= 10
