# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestTrueNorthPerformance from former test_model_truenorth.py

"""Focused suite: TestTrueNorthPerformance from former test_model_truenorth.py."""

from __future__ import annotations

from tests.model_truenorth_support import *  # noqa: F403


class TestTrueNorthPerformance:
    def test_isolation_throughput(self):
        n = TrueNorthNeuron()
        N = 100000
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(50)
        elapsed = time.perf_counter() - t0
        steps_per_s = N / elapsed
        assert steps_per_s > 100000, f"{steps_per_s:.0f} steps/s"

    def test_network_throughput(self):
        pop = Population(TrueNorthNeuron, n=50, label="bench")
        drive = PoissonInput(n=50, rate_hz=500.0, weight=50.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        t0 = time.perf_counter()
        net.run(duration=0.5, dt=0.001, backend="python")
        elapsed = time.perf_counter() - t0
        neuron_steps = 50 * 500
        nsteps_per_s = neuron_steps / elapsed
        assert nsteps_per_s > 5000, f"{nsteps_per_s:.0f} neuron-steps/s"
