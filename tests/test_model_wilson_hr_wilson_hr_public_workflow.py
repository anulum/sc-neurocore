# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestWilsonHRPublicWorkflow from former test_model_wilson_hr.py

"""Focused suite: TestWilsonHRPublicWorkflow from former test_model_wilson_hr.py."""

from __future__ import annotations

from tests.model_wilson_hr_support import *  # noqa: F403

class TestWilsonHRPublicWorkflow:
    """Named workflow contract: Wilson-HR public surface inside the Python simulator."""

    def test_population(self):
        assert Population(WilsonHRNeuron, n=10, label="whr").n == 10

    def test_network_spikes(self):
        pop = Population(WilsonHRNeuron, n=10, label="whr")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=20.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=5.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_projection_wiring(self):
        src = Population(WilsonHRNeuron, n=10, label="src")
        tgt = Population(WilsonHRNeuron, n=10, label="tgt")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=20.0, dt=0.001, seed=42)
        proj = Projection(src, tgt, weight=0.2, probability=1.0, seed=42)
        mon = SpikeMonitor(src)
        net = Network(src, tgt, drive, proj, mon)
        net.run(duration=5.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_analysis(self):
        n = WilsonHRNeuron()
        train = np.array([float(n.step(10.0)) for _ in range(50_000)])
        sc = spike_count(train)
        assert sc >= 4
        intervals = isi(train, dt=0.00005)
        assert len(intervals) >= 3
        rate = firing_rate(train, dt=0.00005)
        assert rate > 0
