# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBertramPipeline from former test_model_bertram_phantom.py

"""Focused suite: TestBertramPipeline from former test_model_bertram_phantom.py."""

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
from sc_neurocore.neurons.models.sc_three_state_phantom import (
    SCThreeStatePhantomBurster as BertramPhantomBurster,
)


class TestBertramPipeline:
    """The BertramPhantomBurster in the network pipeline."""

    def test_population(self) -> None:
        """A population of the model is built."""
        pop = Population(BertramPhantomBurster, n=10, label="bertram")
        assert pop.n == 10

    def test_projection_wiring(self) -> None:
        """Full src→tgt wiring via Projection with SpikeMonitor."""
        src = Population(BertramPhantomBurster, n=5, label="src")
        tgt = Population(BertramPhantomBurster, n=5, label="tgt")
        drive = PoissonInput(n=5, rate_hz=1000.0, weight=200.0, dt=0.001, seed=42)
        proj = Projection(src, tgt, weight=50.0, probability=1.0, seed=42)
        mon_src = SpikeMonitor(src)
        mon_tgt = SpikeMonitor(tgt)
        net = Network(src, tgt, drive, proj, mon_src, mon_tgt)
        net.run(duration=2.0, dt=0.001, backend="python")
        assert mon_src.count > 0

    def test_network_spikes(self) -> None:
        """A driven network runs and is monitored."""
        pop = Population(BertramPhantomBurster, n=10, label="bertram")
        drive = PoissonInput(n=10, rate_hz=1000.0, weight=200.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_spike_trains_extractable(self) -> None:
        """The network spike trains can be read out."""
        pop = Population(BertramPhantomBurster, n=5, label="bertram")
        drive = PoissonInput(n=5, rate_hz=1000.0, weight=200.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        trains = mon.spike_trains
        assert isinstance(trains, dict)

    def test_analysis_spike_count(self) -> None:
        """The analysis counts the spikes of a driven run."""
        n = BertramPhantomBurster()
        train = np.array([float(n.step(200.0)) for _ in range(50_000)])
        sc = spike_count(train)
        assert sc == 1

    def test_analysis_isi(self) -> None:
        """The analysis gives finite inter-spike intervals."""
        n = BertramPhantomBurster()
        train = np.array([float(n.step(200.0)) for _ in range(50_000)])
        intervals = isi(train, dt=0.0005)
        if intervals.size > 0:
            assert np.all(np.isfinite(intervals))
            assert np.all(intervals > 0)

    def test_analysis_firing_rate(self) -> None:
        """The analysis gives the firing rate of a driven run."""
        n = BertramPhantomBurster()
        train = np.array([float(n.step(200.0)) for _ in range(50_000)])
        rate = firing_rate(train, dt=0.0005)
        assert rate > 0

    def test_analysis_cross_validation(self) -> None:
        """spike_count / duration ≈ firing_rate."""
        n = BertramPhantomBurster()
        train = np.array([float(n.step(200.0)) for _ in range(50_000)])
        sc = spike_count(train)
        dt_sim = 0.0005
        duration = len(train) * dt_sim
        rate = firing_rate(train, dt=dt_sim)
        if sc > 0:
            expected_rate = sc / duration
            assert abs(rate - expected_rate) < expected_rate * 0.1
