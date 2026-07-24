# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestAdExPipeline from former test_model_adex.py

"""Focused suite: TestAdExPipeline from former test_model_adex.py."""

from __future__ import annotations

from tests.model_adex_support import *  # noqa: F403


class TestAdExPipeline:
    def test_population(self):
        assert Population(AdExNeuron, n=10, label="adex").n == 10

    def test_network_with_drive(self):
        pop = Population(AdExNeuron, n=10, label="adex")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=500.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=2.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_projection_wiring(self):
        """Projection from source adds current to target. Verify source fires
        and projection object is accepted by Network without error."""
        src = Population(AdExNeuron, n=5, label="src")
        tgt = Population(AdExNeuron, n=5, label="tgt")
        drive = PoissonInput(n=5, rate_hz=500.0, weight=500.0, dt=0.001, seed=42)
        proj = Projection(src, tgt, weight=200.0, probability=1.0, seed=42)
        mon_src = SpikeMonitor(src)
        mon_tgt = SpikeMonitor(tgt)
        # Run without error — Projection is wired into the network graph
        net = Network(src, tgt, drive, proj, mon_src, mon_tgt)
        net.run(duration=2.0, dt=0.001, backend="python")
        assert mon_src.count > 0, "Source should fire"
        # Target may or may not fire depending on projection current magnitude
        # The key test: network accepted the Projection and ran without error

    def test_analysis_pipeline(self):
        n = AdExNeuron()
        train = np.array([float(n.step(500.0)) for _ in range(10000)])
        sc = spike_count(train)
        assert sc >= 10
        isis = isi(train, dt=0.0001)  # dt = 0.1 ms per step
        assert len(isis) >= 5
        rate = firing_rate(train, dt=0.0001)
        assert rate > 0
