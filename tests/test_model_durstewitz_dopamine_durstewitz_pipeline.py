# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDurstewitzPipeline from former test_model_durstewitz_dopamine.py

"""Focused suite: TestDurstewitzPipeline from former test_model_durstewitz_dopamine.py."""

from __future__ import annotations

from tests.model_durstewitz_dopamine_support import *  # noqa: F403

class TestDurstewitzPipeline:
    def test_population(self):
        assert Population(DurstewitzDopamineNeuron, n=5, label="dd").n == 5

    def test_network_spikes(self):
        pop = Population(DurstewitzDopamineNeuron, n=5, label="dd")
        drive = PoissonInput(n=5, rate_hz=200.0, weight=10.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=5.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_projection_wiring(self):
        src = Population(DurstewitzDopamineNeuron, n=5, label="src")
        tgt = Population(DurstewitzDopamineNeuron, n=5, label="tgt")
        drive = PoissonInput(n=5, rate_hz=200.0, weight=10.0, dt=0.001, seed=42)
        proj = Projection(src, tgt, weight=5.0, probability=1.0, seed=42)
        mon = SpikeMonitor(src)
        net = Network(src, tgt, drive, proj, mon)
        net.run(duration=5.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_analysis(self):
        n = DurstewitzDopamineNeuron()
        train = np.array([float(n.step(10.0)) for _ in range(10000)])
        sc = spike_count(train)
        assert sc >= 10
        rate = firing_rate(train, dt=0.001)
        assert rate > 0
