# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestHBPerformance from former test_model_huber_braun.py

"""Focused suite: TestHBPerformance from former test_model_huber_braun.py."""

from __future__ import annotations

from tests.model_huber_braun_support import *  # noqa: F403

class TestHBPerformance:
    def test_isolation_throughput(self):
        n = HuberBraunNeuron()
        N = 50_000
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(50.0)
        elapsed = time.perf_counter() - t0
        rate = N / elapsed
        assert rate > 20_000, f"isolation: {rate:.0f} steps/s"

    def test_network_throughput(self):
        pop = Population(HuberBraunNeuron, n=20, label="bench")
        drive = PoissonInput(n=20, rate_hz=500.0, weight=50.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        t0 = time.perf_counter()
        net.run(duration=0.5, dt=0.001, backend="python")
        elapsed = time.perf_counter() - t0
        neuron_steps = 20 * 500
        rate = neuron_steps / elapsed
        assert rate > 1_000, f"network: {rate:.0f} neuron-steps/s"
