# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestArcanePipeline from former test_model_arcane_neuron.py

"""Focused suite: TestArcanePipeline from former test_model_arcane_neuron.py."""

from __future__ import annotations

from tests.model_arcane_neuron_support import *  # noqa: F403


class TestArcanePipeline:
    def test_population(self):
        assert Population(ArcaneNeuron, n=10, label="arcane").n == 10

    def test_network_spikes(self):
        pop = Population(ArcaneNeuron, n=10, label="arcane")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=3.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=2.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_projection_wiring(self):
        src = Population(ArcaneNeuron, n=5, label="src")
        tgt = Population(ArcaneNeuron, n=5, label="tgt")
        drive = PoissonInput(n=5, rate_hz=500.0, weight=3.0, dt=0.001, seed=42)
        proj = Projection(src, tgt, weight=2.0, probability=1.0, seed=42)
        mon_src = SpikeMonitor(src)
        mon_tgt = SpikeMonitor(tgt)
        net = Network(src, tgt, drive, proj, mon_src, mon_tgt)
        net.run(duration=2.0, dt=0.001, backend="python")
        assert mon_src.count > 0
        assert mon_tgt.count > 0

    def test_analysis(self):
        n = ArcaneNeuron()
        train = np.array([float(n.step(2.0)) for _ in range(5000)])
        sc = spike_count(train)
        assert sc >= 50
        rate = firing_rate(train, dt=0.001)
        assert rate > 0

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = ArcaneNeuron()
            trace = [(n.step(2.0), n.v_fast) for _ in range(200)]
            traces.append(trace)
        assert traces[0] == traces[1]
