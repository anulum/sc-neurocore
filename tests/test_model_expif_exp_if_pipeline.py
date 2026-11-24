# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestExpIFPipeline from former test_model_expif.py

"""Focused suite: TestExpIFPipeline from former test_model_expif.py."""

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
from sc_neurocore.network.projection import (
    Projection,
)
from sc_neurocore.network.stimulus import (
    PoissonInput,
)
from sc_neurocore.neurons.models.expif import (
    ExpIFNeuron,
)


class TestExpIFPipeline:
    """The ExpIFNeuron in the network pipeline."""

    def test_population(self) -> None:
        """A population of the model is built."""
        assert Population(ExpIFNeuron, n=10, label="expif").n == 10

    def test_network_spikes(self) -> None:
        """A driven network runs and is monitored."""
        population = Population(ExpIFNeuron, n=10, label="expif")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=50.0, dt=0.001, seed=42)
        monitor = SpikeMonitor(population)
        network = Network(population, drive, monitor)
        network.run(duration=1.0, dt=0.001, backend="python")
        assert monitor.count > 0

    def test_projection_wiring(self) -> None:
        """A projection wires the source to the target."""
        source = Population(ExpIFNeuron, n=10, label="src")
        target = Population(ExpIFNeuron, n=10, label="tgt")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=50.0, dt=0.001, seed=42)
        projection = Projection(source, target, weight=20.0, probability=1.0, seed=42)
        monitor = SpikeMonitor(source)
        network = Network(source, target, drive, projection, monitor)
        network.run(duration=1.0, dt=0.001, backend="python")
        assert monitor.count > 0

    def test_analysis_pipeline(self) -> None:
        """The analysis pipeline reports a driven run."""
        neuron = ExpIFNeuron()
        train = np.array([float(neuron.step(50.0)) for _ in range(10_000)])
        assert spike_count(train) == 52
        assert len(isi(train, dt=0.00002)) >= 5
        assert firing_rate(train, dt=0.00002) > 0.0
