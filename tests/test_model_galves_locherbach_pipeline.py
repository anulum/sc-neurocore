# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPipeline from former test_model_galves_locherbach.py

"""Focused suite: TestPipeline from former test_model_galves_locherbach.py."""

from __future__ import annotations

from tests.model_galves_locherbach_support import *  # noqa: F403

class TestPipeline:
    def test_population(self):
        assert Population(GalvesLocherbachNeuron, n=10, label="test").n == 10

    def test_network_spikes(self):
        pop = Population(GalvesLocherbachNeuron, n=10, label="test")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=10.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=2.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_projection_wiring(self):
        src = Population(GalvesLocherbachNeuron, n=5, label="src")
        tgt = Population(GalvesLocherbachNeuron, n=5, label="tgt")
        drive = PoissonInput(n=5, rate_hz=500.0, weight=10.0, dt=0.001, seed=42)
        proj = Projection(src, tgt, weight=10.0, probability=1.0, seed=42)
        mon = SpikeMonitor(src)
        net = Network(src, tgt, drive, proj, mon)
        net.run(duration=2.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_analysis(self):
        n = GalvesLocherbachNeuron()
        train = np.array([float(n.step(10.0)) for _ in range(5000)])
        sc = spike_count(train)
        assert sc >= 5
        rate = firing_rate(train, dt=0.001)
        assert rate > 0
