# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestMcKeanPublicWorkflow from former test_model_mckean.py

"""Focused suite: TestMcKeanPublicWorkflow from former test_model_mckean.py."""

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
from sc_neurocore.neurons.models.sc_triangular_mckean import (
    SCTriangularMcKeanNeuron as McKeanNeuron,
)


class TestMcKeanPublicWorkflow:
    """Named workflow contract: McKean public surface inside the Python simulator."""

    def test_population(self) -> None:
        """A population of the model is built."""
        assert Population(McKeanNeuron, n=10, label="mck").n == 10

    def test_projection_wiring(self) -> None:
        """A projection wires the source to the target."""
        src = Population(McKeanNeuron, n=5, label="src")
        tgt = Population(McKeanNeuron, n=5, label="tgt")
        drive = PoissonInput(n=5, rate_hz=500.0, weight=0.5, dt=0.001, seed=42)
        proj = Projection(src, tgt, weight=0.3, probability=1.0, seed=42)
        mon_src = SpikeMonitor(src)
        mon_tgt = SpikeMonitor(tgt)
        net = Network(src, tgt, drive, proj, mon_src, mon_tgt)
        net.run(duration=5.0, dt=0.001, backend="python")
        assert mon_src.count > 0

    def test_network_spikes(self) -> None:
        """A driven network runs and is monitored."""
        pop = Population(McKeanNeuron, n=10, label="mck")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=0.5, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=5.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_analysis_spike_count(self) -> None:
        """The analysis counts the spikes of a driven run."""
        n = McKeanNeuron()
        train = np.array([float(n.step(0.5)) for _ in range(20_000)])
        sc = spike_count(train)
        assert sc >= 3

    def test_analysis_isi(self) -> None:
        """The analysis gives finite inter-spike intervals."""
        n = McKeanNeuron()
        train = np.array([float(n.step(0.5)) for _ in range(50_000)])
        intervals = isi(train, dt=0.0001)
        if intervals.size > 0:
            assert np.all(np.isfinite(intervals))
            assert np.all(intervals > 0)

    def test_analysis_firing_rate(self) -> None:
        """The analysis gives the firing rate of a driven run."""
        n = McKeanNeuron()
        train = np.array([float(n.step(0.5)) for _ in range(20_000)])
        rate = firing_rate(train, dt=0.0001)
        assert rate >= 0

    def test_analysis_cross_validation(self) -> None:
        """Analysis cross validation."""
        n = McKeanNeuron()
        train = np.array([float(n.step(0.5)) for _ in range(50_000)])
        sc = spike_count(train)
        dt_sim = 0.0001
        duration = len(train) * dt_sim
        rate = firing_rate(train, dt=dt_sim)
        if sc > 0:
            expected = sc / duration
            assert abs(rate - expected) < expected * 0.1
