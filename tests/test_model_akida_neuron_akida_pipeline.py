# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestAkidaPipeline from former test_model_akida_neuron.py

"""Focused suite: TestAkidaPipeline from former test_model_akida_neuron.py."""

from __future__ import annotations

from tests.model_akida_neuron_support import *  # noqa: F403

class TestAkidaPipeline:
    def test_population(self):
        assert Population(AkidaNeuron, n=10, label="akida").n == 10

    def test_projection_wiring(self):
        src = Population(AkidaNeuron, n=5, label="src")
        tgt = Population(AkidaNeuron, n=5, label="tgt")
        drive = PoissonInput(n=5, rate_hz=500.0, weight=100.0, dt=0.001, seed=42)
        proj = Projection(src, tgt, weight=50.0, probability=1.0, seed=42)
        mon_src = SpikeMonitor(src)
        net = Network(src, tgt, drive, proj, mon_src)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert isinstance(mon_src.count, int)

    def test_network_spikes(self):
        pop = Population(AkidaNeuron, n=10, label="akida")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=100.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        # Single-spike model: at most n spikes total
        assert mon.count <= 10

    def test_analysis(self):
        n = AkidaNeuron()
        train = np.array([float(n.step(100)) for _ in range(100)])
        sc = spike_count(train)
        # Single spike model
        assert sc <= 1
