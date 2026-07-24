# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestStochasticIFPipeline from former test_model_stochastic_if.py

"""Focused suite: TestStochasticIFPipeline from former test_model_stochastic_if.py."""

from __future__ import annotations

from tests.model_stochastic_if_support import *  # noqa: F403


class TestStochasticIFPipeline:
    def test_population(self):
        assert Population(StochasticIFNeuron, n=10, label="sif").n == 10

    def test_network_with_drive(self):
        pop = Population(StochasticIFNeuron, n=10, label="sif")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=25.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_projection_affects_target(self):
        """Verify projection wiring by comparing target spikes with/without projection.

        Both target populations get the same subthreshold drive.
        Only one gets a projection from a firing source.
        """
        src = Population(StochasticIFNeuron, n=20, label="src")
        tgt_with = Population(StochasticIFNeuron, n=20, label="tgt_proj")
        tgt_without = Population(StochasticIFNeuron, n=20, label="tgt_noproj")
        drive_src = PoissonInput(n=20, rate_hz=500.0, weight=30.0, dt=0.001, seed=42)
        drive_tgt1 = PoissonInput(n=20, rate_hz=200.0, weight=15.0, dt=0.001, seed=99)
        drive_tgt2 = PoissonInput(n=20, rate_hz=200.0, weight=15.0, dt=0.001, seed=99)
        proj = Projection(src, tgt_with, weight=50.0, probability=1.0, seed=42)
        mon_src = SpikeMonitor(src)
        mon_with = SpikeMonitor(tgt_with)
        mon_without = SpikeMonitor(tgt_without)
        net_with = Network(src, tgt_with, drive_src, drive_tgt1, proj, mon_src, mon_with)
        net_without = Network(tgt_without, drive_tgt2, mon_without)
        net_with.run(duration=2.0, dt=0.001, backend="python")
        net_without.run(duration=2.0, dt=0.001, backend="python")
        assert mon_src.count > 0, "Source should fire"
        # Target with projection should fire at least as much as without
        # (projection adds excitatory input)
        assert mon_with.count >= mon_without.count

    def test_analysis_pipeline(self):
        n = StochasticIFNeuron()
        train = np.array([float(n.step(25.0)) for _ in range(50000)])
        sc = spike_count(train)
        assert sc >= 50
        isis = isi(train, dt=0.001)
        assert len(isis) >= 10
        rate = firing_rate(train, dt=0.001)
        assert rate > 0
        duration = 50000 * 0.001
        assert abs(rate - sc / duration) < 10.0
