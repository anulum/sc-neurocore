# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestEPropALIFPipeline from former test_model_e_prop_alif.py

"""Focused suite: TestEPropALIFPipeline from former test_model_e_prop_alif.py."""

from __future__ import annotations

from tests.model_e_prop_alif_support import *  # noqa: F403

class TestEPropALIFPipeline:
    def test_population(self):
        assert Population(EPropALIFNeuron, n=10, label="eprop").n == 10

    def test_network_spikes(self):
        pop = Population(EPropALIFNeuron, n=10, label="eprop")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=0.5, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_projection_wiring(self):
        src = Population(EPropALIFNeuron, n=10, label="src")
        tgt = Population(EPropALIFNeuron, n=10, label="tgt")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=0.5, dt=0.001, seed=42)
        proj = Projection(src, tgt, weight=0.3, probability=1.0, seed=42)
        mon = SpikeMonitor(src)
        net = Network(src, tgt, drive, proj, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_analysis(self):
        n = EPropALIFNeuron()
        train = np.array([float(n.step(0.2)) for _ in range(5000)])
        sc = spike_count(train)
        assert sc >= 50
        rate = firing_rate(train, dt=0.001)
        assert rate > 0
