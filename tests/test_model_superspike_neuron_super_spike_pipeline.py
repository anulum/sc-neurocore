# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSuperSpikePipeline from former test_model_superspike_neuron.py

"""Focused suite: TestSuperSpikePipeline from former test_model_superspike_neuron.py."""

from __future__ import annotations

from tests.model_superspike_neuron_support import *  # noqa: F403

class TestSuperSpikePipeline:
    def test_population(self):
        assert Population(SuperSpikeNeuron, n=10, label="ss").n == 10

    def test_network_with_drive(self):
        pop = Population(SuperSpikeNeuron, n=10, label="ss")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=0.5, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_projection_wiring(self):
        src = Population(SuperSpikeNeuron, n=10, label="src")
        tgt = Population(SuperSpikeNeuron, n=10, label="tgt")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=0.5, dt=0.001, seed=42)
        proj = Projection(src, tgt, weight=0.5, probability=1.0, seed=42)
        mon = SpikeMonitor(src)
        net = Network(src, tgt, drive, proj, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_analysis_pipeline(self):
        n = SuperSpikeNeuron()
        train = np.array([float(n.step(0.2)) for _ in range(10000)])
        sc = spike_count(train)
        assert sc >= 50
        rate = firing_rate(train, dt=0.001)
        assert rate > 0

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = SuperSpikeNeuron()
            trace = [(n.step(0.2), n.v, n.trace) for _ in range(200)]
            traces.append(trace)
        assert traces[0] == traces[1]
