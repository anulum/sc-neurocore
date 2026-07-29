# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCSPerformance from former test_model_connor_stevens.py

"""Focused suite: TestCSPerformance from former test_model_connor_stevens.py."""

from __future__ import annotations

from tests.model_connor_stevens_support import *  # noqa: F403
from tests.performance_guard import assert_load_tolerant_throughput


class TestCSPerformance:
    def test_isolation_throughput(self):
        n = ConnorStevensNeuron()
        N = 200
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(20.0)
        elapsed = time.perf_counter() - t0
        rate = N / elapsed
        # 100 sub-steps × HH → ~500 steps/s
        assert_load_tolerant_throughput(
            label="Connor-Stevens isolation",
            observed_per_second=rate,
            strict_minimum_per_second=50.0,
        )

    def test_network_throughput(self):
        pop = Population(ConnorStevensNeuron, n=5, label="bench")
        drive = PoissonInput(n=5, rate_hz=500.0, weight=20.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        t0 = time.perf_counter()
        net.run(duration=0.1, dt=0.001, backend="python")
        elapsed = time.perf_counter() - t0
        neuron_steps = 5 * 100
        rate = neuron_steps / elapsed
        assert_load_tolerant_throughput(
            label="Connor-Stevens network",
            observed_per_second=rate,
            strict_minimum_per_second=10.0,
        )
