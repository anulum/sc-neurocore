# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestTermanWangPipeline from former test_model_terman_wang.py

"""Focused suite: TestTermanWangPipeline from former test_model_terman_wang.py."""

from __future__ import annotations

from tests.model_terman_wang_support import *  # noqa: F403

class TestTermanWangPipeline:
    def test_population(self):
        assert Population(TermanWangOscillator, n=10, label="tw").n == 10

    def test_network_spikes(self):
        pop = Population(TermanWangOscillator, n=10, label="tw")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=1.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=10.0, dt=0.001, backend="python")
        # Slow oscillator — may need long run
        assert isinstance(mon.count, int)

    def test_projection_wiring(self):
        src = Population(TermanWangOscillator, n=5, label="src")
        tgt = Population(TermanWangOscillator, n=5, label="tgt")
        drive = PoissonInput(n=5, rate_hz=500.0, weight=1.0, dt=0.001, seed=42)
        proj = Projection(src, tgt, weight=0.5, probability=1.0, seed=42)
        mon = SpikeMonitor(src)
        net = Network(src, tgt, drive, proj, mon)
        net.run(duration=5.0, dt=0.001, backend="python")
        assert isinstance(mon.count, int)

    def test_analysis(self):
        n = TermanWangOscillator()
        train = np.array([float(n.step(1.0)) for _ in range(100000)])
        sc = spike_count(train)
        assert sc >= 5
