# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestGFSPipeline from former test_model_golomb_fs.py

"""Focused suite: TestGFSPipeline from former test_model_golomb_fs.py."""

from __future__ import annotations

from tests.model_golomb_fs_support import *  # noqa: F403

class TestGFSPipeline:
    def test_population(self):
        assert Population(GolombFSNeuron, n=10, label="gfs").n == 10

    def test_projection_wiring(self):
        src = Population(GolombFSNeuron, n=3, label="src")
        tgt = Population(GolombFSNeuron, n=3, label="tgt")
        drive = PoissonInput(n=3, rate_hz=500.0, weight=5.0, dt=0.001, seed=42)
        proj = Projection(src, tgt, weight=2.0, probability=1.0, seed=42)
        mon_src = SpikeMonitor(src)
        net = Network(src, tgt, drive, proj, mon_src)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert isinstance(mon_src.count, int)

    def test_network_spikes(self):
        pop = Population(GolombFSNeuron, n=5, label="gfs")
        drive = PoissonInput(n=5, rate_hz=500.0, weight=5.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=2.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_analysis_spike_count(self):
        n = GolombFSNeuron()
        train = np.array([float(n.step(5.0)) for _ in range(5000)])
        sc = spike_count(train)
        assert sc >= 5

    def test_analysis_isi(self):
        n = GolombFSNeuron()
        train = np.array([float(n.step(5.0)) for _ in range(5000)])
        intervals = isi(train, dt=0.001)
        if intervals.size > 0:
            assert np.all(np.isfinite(intervals))

    def test_analysis_firing_rate(self):
        n = GolombFSNeuron()
        train = np.array([float(n.step(5.0)) for _ in range(5000)])
        rate = firing_rate(train, dt=0.001)
        assert rate > 0
