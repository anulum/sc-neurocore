# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLBPipeline from former test_model_larter_breakspear.py

"""Focused suite: TestLBPipeline from former test_model_larter_breakspear.py."""

from __future__ import annotations

from tests.model_larter_breakspear_support import *  # noqa: F403

class TestLBPipeline:
    def test_population(self):
        assert Population(LarterBreakspearNeuron, n=5, label="lb").n == 5

    def test_projection_wiring(self):
        src = Population(LarterBreakspearNeuron, n=5, label="src")
        tgt = Population(LarterBreakspearNeuron, n=5, label="tgt")
        drive = PoissonInput(n=5, rate_hz=500.0, weight=0.5, dt=0.001, seed=42)
        proj = Projection(src, tgt, weight=0.3, probability=1.0, seed=42)
        mon_src = SpikeMonitor(src)
        net = Network(src, tgt, drive, proj, mon_src)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert isinstance(mon_src.count, int)

    def test_network_runs(self):
        pop = Population(LarterBreakspearNeuron, n=10, label="lb")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=0.5, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert isinstance(mon.count, int)
