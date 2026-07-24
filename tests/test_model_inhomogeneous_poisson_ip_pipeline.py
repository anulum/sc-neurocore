# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestIPPipeline from former test_model_inhomogeneous_poisson.py

"""Focused suite: TestIPPipeline from former test_model_inhomogeneous_poisson.py."""

from __future__ import annotations

from tests.model_inhomogeneous_poisson_support import *  # noqa: F403


class TestIPPipeline:
    def test_population(self):
        assert Population(InhomogeneousPoissonNeuron, n=10, label="ip").n == 10

    def test_projection_wiring(self):
        src = Population(InhomogeneousPoissonNeuron, n=5, label="src")
        tgt = Population(InhomogeneousPoissonNeuron, n=5, label="tgt")
        drive = PoissonInput(n=5, rate_hz=500.0, weight=100.0, dt=0.001, seed=42)
        proj = Projection(src, tgt, weight=50.0, probability=1.0, seed=42)
        mon_src = SpikeMonitor(src)
        net = Network(src, tgt, drive, proj, mon_src)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert isinstance(mon_src.count, int)

    def test_network_spikes(self):
        pop = Population(InhomogeneousPoissonNeuron, n=10, label="ip")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=100.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert isinstance(mon.count, int)

    def test_analysis_spike_count(self):
        n = InhomogeneousPoissonNeuron()
        train = np.array([float(n.step(100.0)) for _ in range(10_000)])
        sc = spike_count(train)
        assert sc >= 100

    def test_analysis_isi(self):
        n = InhomogeneousPoissonNeuron()
        train = np.array([float(n.step(100.0)) for _ in range(10_000)])
        intervals = isi(train, dt=0.001)
        if intervals.size > 0:
            assert np.all(np.isfinite(intervals))

    def test_analysis_firing_rate(self):
        n = InhomogeneousPoissonNeuron()
        train = np.array([float(n.step(100.0)) for _ in range(10_000)])
        rate = firing_rate(train, dt=0.001)
        assert rate > 0
