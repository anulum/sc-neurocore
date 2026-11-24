# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSTGPipeline from former test_model_marder_stg.py

"""Focused suite: TestSTGPipeline from former test_model_marder_stg.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.analysis.spike_stats.basic import (
    firing_rate,
    isi,
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
from sc_neurocore.neurons.models.marder_stg import (
    MarderSTGNeuron,
)


class TestSTGPipeline:
    """The MarderSTGNeuron in the network pipeline."""

    def test_population(self) -> None:
        """A population of the model is built."""
        assert Population(MarderSTGNeuron, n=10, label="stg").n == 10

    def test_network_spikes(self) -> None:
        """A driven network runs and is monitored."""
        pop = Population(MarderSTGNeuron, n=10, label="stg")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=2.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_analysis_spike_count(self) -> None:
        """The analysis counts the spikes of a driven run."""
        n = MarderSTGNeuron()
        train = np.array([float(n.step(0.0)) for _ in range(50_000)])
        assert spike_count(train) >= 10
        assert spike_count(train) == int(train.sum())

    def test_analysis_firing_rate(self) -> None:
        """The analysis gives the firing rate of a driven run."""
        n = MarderSTGNeuron()
        train = np.array([float(n.step(0.0)) for _ in range(50_000)])
        intervals = isi(train, dt=0.00005)
        if intervals.size > 0:
            assert np.all(intervals > 0)
        assert firing_rate(train, dt=0.00005) > 0
