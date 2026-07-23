# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestWilsonHRPerformance from former test_model_wilson_hr.py

"""Focused suite: TestWilsonHRPerformance from former test_model_wilson_hr.py."""

from __future__ import annotations

from tests.model_wilson_hr_support import *  # noqa: F403

class TestWilsonHRPerformance:
    def test_isolation_throughput(self):
        n = WilsonHRNeuron()
        steps = 50_000
        t0 = time.perf_counter()
        for _ in range(steps):
            n.step(0.3)
        elapsed = time.perf_counter() - t0
        assert steps / elapsed > 10_000

    def test_network_throughput(self):
        pop = Population(WilsonHRNeuron, n=50, label="bench")
        drive = PoissonInput(n=50, rate_hz=500.0, weight=0.3, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        t0 = time.perf_counter()
        net.run(duration=0.5, dt=0.001, backend="python")
        elapsed = time.perf_counter() - t0
        assert 50 * 500 / elapsed > 5_000
