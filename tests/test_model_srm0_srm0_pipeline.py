# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSRM0Pipeline from former test_model_srm0.py

"""Focused suite: TestSRM0Pipeline from former test_model_srm0.py."""

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
from sc_neurocore.neurons.models.srm0 import (
    SRM0Neuron,
)


class TestSRM0Pipeline:
    """The SRM0Neuron in the network pipeline."""

    def test_population(self) -> None:
        """A population of the model is built."""
        assert Population(SRM0Neuron, n=10, label="srm0").n == 10

    def test_network_spikes(self) -> None:
        """A driven network runs and is monitored."""
        pop = Population(SRM0Neuron, n=10, label="srm0")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=5.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_projection_wiring(self) -> None:
        """A projection wires the source to the target."""
        src = Population(SRM0Neuron, n=10, label="src")
        tgt = Population(SRM0Neuron, n=10, label="tgt")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=5.0, dt=0.001, seed=42)
        proj = Projection(src, tgt, weight=3.0, probability=1.0, seed=42)
        mon = SpikeMonitor(src)
        net = Network(src, tgt, drive, proj, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_analysis(self) -> None:
        """The analysis reports spikes and a rate for a driven run."""
        n = SRM0Neuron()
        train = np.array([float(n.step(5.0)) for _ in range(10000)])
        sc = spike_count(train)
        assert sc >= 50
        isis = isi(train, dt=0.001)
        assert len(isis) >= 10
        rate = firing_rate(train, dt=0.001)
        assert rate > 0

    def test_deterministic(self) -> None:
        """The same inputs give the same trace."""
        traces = []
        for _ in range(2):
            n = SRM0Neuron()
            trace = [(n.step(3.0), n.v) for _ in range(200)]
            traces.append(trace)
        assert traces[0] == traces[1]
