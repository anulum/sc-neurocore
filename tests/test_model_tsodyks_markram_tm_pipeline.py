# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestTMPipeline from former test_model_tsodyks_markram.py

"""Focused suite: TestTMPipeline from former test_model_tsodyks_markram.py."""

from __future__ import annotations

from tests.model_tsodyks_markram_support import *  # noqa: F403

class TestTMPipeline:
    def test_population(self):
        assert Population(TsodyksMarkramNeuron, n=10, label="tm").n == 10

    def test_network_spikes(self):
        pop = Population(TsodyksMarkramNeuron, n=10, label="tm")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=50.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=2.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_analysis(self):
        n = TsodyksMarkramNeuron()
        train = np.array([float(n.step(30.0)) for _ in range(10000)])
        sc = spike_count(train)
        assert sc >= 10
        rate = firing_rate(train, dt=0.0001)
        assert rate > 0

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = TsodyksMarkramNeuron()
            trace = [(n.step(20.0), n.v, n.x, n.u) for _ in range(200)]
            traces.append(trace)
        assert traces[0] == traces[1]
