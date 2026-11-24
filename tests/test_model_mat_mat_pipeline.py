# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestMATPipeline from former test_model_mat.py

"""Focused suite: TestMATPipeline from former test_model_mat.py."""

from __future__ import annotations

from sc_neurocore.network.network import Network
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.neurons.models.sc_resetting_mat import SCResettingMATNeuron
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.analysis.spike_stats.basic import firing_rate
from sc_neurocore.analysis.spike_stats.basic import isi
import numpy as np
from sc_neurocore.analysis.spike_stats.basic import spike_count


class TestMATPipeline:
    """The resetting MAT neuron in the network pipeline."""

    def test_population(self) -> None:
        """A population of ten neurons is built."""
        assert Population(SCResettingMATNeuron, n=10, label="mat").n == 10

    def test_projection_wiring(self) -> None:
        """A driven projection makes the source population spike."""
        src = Population(SCResettingMATNeuron, n=5, label="src")
        tgt = Population(SCResettingMATNeuron, n=5, label="tgt")
        drive = PoissonInput(n=5, rate_hz=500.0, weight=30.0, dt=0.001, seed=42)
        proj = Projection(src, tgt, weight=10.0, probability=1.0, seed=42)
        mon_src = SpikeMonitor(src)
        mon_tgt = SpikeMonitor(tgt)
        net = Network(src, tgt, drive, proj, mon_src, mon_tgt)
        net.run(duration=2.0, dt=0.001, backend="python")
        assert mon_src.count > 0

    def test_network_spikes(self) -> None:
        """A driven network spikes."""
        pop = Population(SCResettingMATNeuron, n=10, label="mat")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=30.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_analysis_spike_count(self) -> None:
        """A driven run gives at least 20 spikes."""
        n = SCResettingMATNeuron()
        train = np.array([float(n.step(30.0)) for _ in range(5000)])
        sc = spike_count(train)
        assert sc >= 20

    def test_analysis_isi(self) -> None:
        """The inter-spike intervals are finite and positive."""
        n = SCResettingMATNeuron()
        train = np.array([float(n.step(30.0)) for _ in range(5000)])
        intervals = isi(train, dt=0.001)
        if intervals.size > 0:
            assert np.all(np.isfinite(intervals))
            assert np.all(intervals > 0)

    def test_analysis_firing_rate(self) -> None:
        """The firing rate of a driven run is positive."""
        n = SCResettingMATNeuron()
        train = np.array([float(n.step(30.0)) for _ in range(5000)])
        rate = firing_rate(train, dt=0.001)
        assert rate > 0

    def test_analysis_cross_validation(self) -> None:
        """The firing rate agrees with the spike count within ten percent."""
        n = SCResettingMATNeuron()
        train = np.array([float(n.step(30.0)) for _ in range(5000)])
        sc = spike_count(train)
        dt_sim = 0.001
        duration = len(train) * dt_sim
        rate = firing_rate(train, dt=dt_sim)
        if sc > 0:
            expected = sc / duration
            assert abs(rate - expected) < expected * 0.1
