# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLoihiCUBAPipeline from former test_model_loihi_cuba.py

"""Focused suite: TestLoihiCUBAPipeline from former test_model_loihi_cuba.py."""

from __future__ import annotations

from tests.model_loihi_cuba_support import *  # noqa: F403


class TestLoihiCUBAPipeline:
    def test_population(self):
        assert Population(LoihiCUBANeuron, n=10, label="cuba").n == 10

    def test_projection_wiring(self):
        src = Population(LoihiCUBANeuron, n=5, label="src")
        tgt = Population(LoihiCUBANeuron, n=5, label="tgt")
        drive = PoissonInput(n=5, rate_hz=500.0, weight=200.0, dt=0.001, seed=42)
        proj = Projection(src, tgt, weight=100.0, probability=1.0, seed=42)
        mon_src = SpikeMonitor(src)
        net = Network(src, tgt, drive, proj, mon_src)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon_src.count > 0

    def test_network_spikes(self):
        pop = Population(LoihiCUBANeuron, n=10, label="cuba")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=200.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_analysis(self):
        n = LoihiCUBANeuron()
        train = np.array([float(n.step(200)) for _ in range(5000)])
        assert spike_count(train) >= 20
