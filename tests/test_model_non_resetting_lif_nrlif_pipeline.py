# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestNRLIFPipeline from former test_model_non_resetting_lif.py

"""Focused suite: TestNRLIFPipeline from former test_model_non_resetting_lif.py."""

from __future__ import annotations

from tests.model_non_resetting_lif_support import *  # noqa: F403

class TestNRLIFPipeline:
    def test_population(self):
        assert Population(NonResettingLIFNeuron, n=10, label="nrlif").n == 10

    def test_projection_wiring(self):
        src = Population(NonResettingLIFNeuron, n=5, label="src")
        tgt = Population(NonResettingLIFNeuron, n=5, label="tgt")
        drive = PoissonInput(n=5, rate_hz=1000.0, weight=200.0, dt=0.001, seed=42)
        proj = Projection(src, tgt, weight=10.0, probability=1.0, seed=42)
        mon_src = SpikeMonitor(src)
        mon_tgt = SpikeMonitor(tgt)
        net = Network(src, tgt, drive, proj, mon_src, mon_tgt)
        net.run(duration=5.0, dt=0.001, backend="python")
        assert mon_src.count > 0

    def test_network_spikes(self):
        pop = Population(NonResettingLIFNeuron, n=10, label="nrlif")
        drive = PoissonInput(n=10, rate_hz=1000.0, weight=200.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=5.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_analysis_spike_count(self):
        n = NonResettingLIFNeuron()
        train = np.array([float(n.step(20.0)) for _ in range(5000)])
        sc = spike_count(train)
        assert sc >= 3

    def test_analysis_isi(self):
        n = NonResettingLIFNeuron()
        train = np.array([float(n.step(20.0)) for _ in range(10_000)])
        intervals = isi(train, dt=0.0001)
        if intervals.size > 0:
            assert np.all(np.isfinite(intervals))
            assert np.all(intervals > 0)

    def test_analysis_firing_rate(self):
        n = NonResettingLIFNeuron()
        train = np.array([float(n.step(20.0)) for _ in range(5000)])
        rate = firing_rate(train, dt=0.0001)
        assert rate >= 0

    def test_analysis_cross_validation(self):
        n = NonResettingLIFNeuron()
        train = np.array([float(n.step(20.0)) for _ in range(10_000)])
        sc = spike_count(train)
        dt_sim = 0.0001
        duration = len(train) * dt_sim
        rate = firing_rate(train, dt=dt_sim)
        if sc > 0:
            expected = sc / duration
            assert abs(rate - expected) < expected * 0.1
