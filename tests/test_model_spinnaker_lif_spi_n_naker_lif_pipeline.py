# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSpiNNakerLIFPipeline from former test_model_spinnaker_lif.py

"""Focused suite: TestSpiNNakerLIFPipeline from former test_model_spinnaker_lif.py."""

from __future__ import annotations

from tests.model_spinnaker_lif_support import *  # noqa: F403


class TestSpiNNakerLIFPipeline:
    def test_population(self):
        assert Population(SpiNNakerLIFNeuron, n=10, label="snlif").n == 10

    def test_network_spikes(self):
        pop = Population(SpiNNakerLIFNeuron, n=10, label="snlif")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=50.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_projection_wiring(self):
        src = Population(SpiNNakerLIFNeuron, n=10, label="src")
        tgt = Population(SpiNNakerLIFNeuron, n=10, label="tgt")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=50.0, dt=0.001, seed=42)
        proj = Projection(src, tgt, weight=25.0, probability=1.0, seed=42)
        mon = SpikeMonitor(src)
        net = Network(src, tgt, drive, proj, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_analysis(self):
        n = SpiNNakerLIFNeuron()
        train = np.array([float(n.step(25.0)) for _ in range(5000)])
        assert spike_count(train) >= 10
        assert firing_rate(train, dt=0.001) > 0

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = SpiNNakerLIFNeuron()
            trace = [(n.step(25.0), n.v) for _ in range(200)]
            traces.append(trace)
        assert traces[0] == traces[1]
