# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCSPipeline from former test_model_connor_stevens.py

"""Focused suite: TestCSPipeline from former test_model_connor_stevens.py."""

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
from sc_neurocore.neurons.models.connor_stevens import (
    ConnorStevensNeuron,
)


class TestCSPipeline:
    """The ConnorStevensNeuron in the network pipeline."""

    def test_population(self) -> None:
        """A population of the model is built."""
        assert Population(ConnorStevensNeuron, n=5, label="cs").n == 5

    def test_projection_wiring(self) -> None:
        """A projection wires the source to the target."""
        src = Population(ConnorStevensNeuron, n=3, label="src")
        tgt = Population(ConnorStevensNeuron, n=3, label="tgt")
        drive = PoissonInput(n=3, rate_hz=500.0, weight=20.0, dt=0.001, seed=42)
        proj = Projection(src, tgt, weight=10.0, probability=1.0, seed=42)
        mon_src = SpikeMonitor(src)
        net = Network(src, tgt, drive, proj, mon_src)
        net.run(duration=0.5, dt=0.001, backend="python")
        assert isinstance(mon_src.count, int)

    def test_network_spikes(self) -> None:
        """A driven network runs and is monitored."""
        pop = Population(ConnorStevensNeuron, n=5, label="cs")
        drive = PoissonInput(n=5, rate_hz=500.0, weight=20.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert isinstance(mon.count, int)

    def test_analysis_spike_count(self) -> None:
        """The analysis counts the spikes of a driven run."""
        n = ConnorStevensNeuron()
        train = np.array([float(n.step(20.0)) for _ in range(500)])
        sc = spike_count(train)
        assert sc >= 10

    def test_analysis_isi(self) -> None:
        """The analysis gives finite inter-spike intervals."""
        n = ConnorStevensNeuron()
        train = np.array([float(n.step(20.0)) for _ in range(500)])
        intervals = isi(train, dt=0.001)
        if intervals.size > 0:
            assert np.all(np.isfinite(intervals))

    def test_analysis_firing_rate(self) -> None:
        """The analysis gives the firing rate of a driven run."""
        n = ConnorStevensNeuron()
        train = np.array([float(n.step(20.0)) for _ in range(500)])
        rate = firing_rate(train, dt=0.001)
        assert rate > 0
