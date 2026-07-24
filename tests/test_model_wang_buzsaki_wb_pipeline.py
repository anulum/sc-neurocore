# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestWBPipeline from former test_model_wang_buzsaki.py

"""Focused suite: TestWBPipeline from former test_model_wang_buzsaki.py."""

from __future__ import annotations

from tests.model_wang_buzsaki_support import *  # noqa: F403


class TestWBPipeline:
    def test_population(self):
        assert Population(WangBuzsakiNeuron, n=10, label="wb").n == 10

    def test_network_with_drive(self):
        pop = Population(WangBuzsakiNeuron, n=10, label="wb")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=2.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_projection_affects_target(self):
        src = Population(WangBuzsakiNeuron, n=10, label="src")
        tgt_with = Population(WangBuzsakiNeuron, n=10, label="tgt_w")
        tgt_without = Population(WangBuzsakiNeuron, n=10, label="tgt_wo")
        drive_src = PoissonInput(n=10, rate_hz=500.0, weight=2.0, dt=0.001, seed=42)
        drive_tgt1 = PoissonInput(n=10, rate_hz=100.0, weight=0.5, dt=0.001, seed=99)
        drive_tgt2 = PoissonInput(n=10, rate_hz=100.0, weight=0.5, dt=0.001, seed=99)
        proj = Projection(src, tgt_with, weight=2.0, probability=1.0, seed=42)
        mon_src = SpikeMonitor(src)
        mon_with = SpikeMonitor(tgt_with)
        mon_without = SpikeMonitor(tgt_without)
        net_with = Network(src, tgt_with, drive_src, drive_tgt1, proj, mon_src, mon_with)
        net_without = Network(tgt_without, drive_tgt2, mon_without)
        net_with.run(duration=1.0, dt=0.001, backend="python")
        net_without.run(duration=1.0, dt=0.001, backend="python")
        assert mon_src.count > 0
        assert mon_with.count >= mon_without.count

    def test_analysis_pipeline(self):
        n = WangBuzsakiNeuron()
        train = np.array([float(n.step(2.0)) for _ in range(20000)])
        sc = spike_count(train)
        assert sc >= 100
        isis = isi(train, dt=0.0005)  # each step = 0.5 ms
        assert len(isis) >= 10
        rate = firing_rate(train, dt=0.0005)
        assert rate > 0
