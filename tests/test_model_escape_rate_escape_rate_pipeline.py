# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestEscapeRatePipeline from former test_model_escape_rate.py

"""Focused suite: TestEscapeRatePipeline from former test_model_escape_rate.py."""

from __future__ import annotations

from tests.model_escape_rate_support import *  # noqa: F403

class TestEscapeRatePipeline:
    def test_population(self):
        assert Population(EscapeRateNeuron, n=10, label="esc").n == 10

    def test_network_spikes(self):
        pop = Population(EscapeRateNeuron, n=20, label="esc")
        drive = PoissonInput(n=20, rate_hz=500.0, weight=50.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_projection_wiring(self):
        """Projection accepted by Network — source fires, graph valid."""
        src = Population(EscapeRateNeuron, n=10, label="src")
        tgt = Population(EscapeRateNeuron, n=10, label="tgt")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=50.0, dt=0.001, seed=42)
        proj = Projection(src, tgt, weight=50.0, probability=1.0, seed=42)
        mon_src = SpikeMonitor(src)
        net = Network(src, tgt, drive, proj, mon_src)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon_src.count > 0

    def test_analysis_pipeline(self):
        n = EscapeRateNeuron()
        train = np.array([float(n.step(40.0)) for _ in range(50000)])
        sc = spike_count(train)
        assert sc >= 50
        isis = isi(train, dt=0.001)
        assert len(isis) >= 10
        rate = firing_rate(train, dt=0.001)
        assert rate > 0
        duration = 50000 * 0.001
        assert abs(rate - sc / duration) < 100.0
