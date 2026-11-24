# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBrunelWangPipeline from former test_model_brunel_wang.py

"""Focused suite: TestBrunelWangPipeline from former test_model_brunel_wang.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.analysis.spike_stats.basic import (
    firing_rate,
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
from sc_neurocore.neurons.models.brunel_wang import (
    BrunelWangNeuron,
)


class TestBrunelWangPipeline:
    """The BrunelWangNeuron in the network pipeline."""

    def test_population(self) -> None:
        """A population of the model is built."""
        assert Population(BrunelWangNeuron, n=10, label="bw").n == 10

    def test_network_spikes(self) -> None:
        """A driven network runs and is monitored."""
        pop = Population(BrunelWangNeuron, n=10, label="bw")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=2.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_projection_wiring(self) -> None:
        """A projection wires the source to the target."""
        src = Population(BrunelWangNeuron, n=5, label="src")
        tgt = Population(BrunelWangNeuron, n=5, label="tgt")
        drive = PoissonInput(n=5, rate_hz=500.0, weight=2.0, dt=0.001, seed=42)
        proj = Projection(src, tgt, weight=1.0, probability=1.0, seed=42)
        mon = SpikeMonitor(src)
        net = Network(src, tgt, drive, proj, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_analysis(self) -> None:
        """The analysis reports spikes and a rate for a driven run."""
        n = BrunelWangNeuron()
        train = np.array([float(n.step(1.0)) for _ in range(10000)])
        sc = spike_count(train)
        assert sc >= 50
        rate = firing_rate(train, dt=0.001)
        assert rate > 0
