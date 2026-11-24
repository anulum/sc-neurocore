# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestTrueNorthPipeline from former test_model_truenorth.py

"""Focused suite: TestTrueNorthPipeline from former test_model_truenorth.py."""

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
from sc_neurocore.neurons.models.truenorth import (
    TrueNorthNeuron,
)


class TestTrueNorthPipeline:
    """The TrueNorthNeuron in the network pipeline."""

    def test_population(self) -> None:
        """A population of the model is built."""
        assert Population(TrueNorthNeuron, n=10, label="tn").n == 10

    def test_network_with_drive(self) -> None:
        """A Poisson-driven population produces spikes."""
        pop = Population(TrueNorthNeuron, n=10, label="tn")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=50.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_projection_wiring(self) -> None:
        """A projection wires the source to the target."""
        src = Population(TrueNorthNeuron, n=10, label="src")
        tgt = Population(TrueNorthNeuron, n=10, label="tgt")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=50.0, dt=0.001, seed=42)
        proj = Projection(src, tgt, weight=50.0, probability=1.0, seed=42)
        mon_src = SpikeMonitor(src)
        mon_tgt = SpikeMonitor(tgt)
        net = Network(src, tgt, drive, proj, mon_src, mon_tgt)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon_src.count > 0

    def test_analysis_pipeline(self) -> None:
        """The analysis pipeline reports a driven run."""
        n = TrueNorthNeuron()
        train = np.array([float(n.step(50)) for _ in range(10000)])
        sc = spike_count(train)
        assert sc >= 100
        isis = isi(train, dt=0.001)
        assert len(isis) >= 10
        rate = firing_rate(train, dt=0.001)
        assert rate > 0
        duration = 10000 * 0.001
        assert abs(rate - sc / duration) < 10.0

    def test_deterministic(self) -> None:
        """The same inputs give the same trace."""
        traces = []
        for _ in range(2):
            n = TrueNorthNeuron()
            trace = [(n.step(50), n.v) for _ in range(200)]
            traces.append(trace)
        assert traces[0] == traces[1]
