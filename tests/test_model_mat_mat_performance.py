# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestMATPerformance from former test_model_mat.py

"""Focused suite: TestMATPerformance from former test_model_mat.py."""

from __future__ import annotations

from tests.model_mat_support import *  # noqa: F403

class TestMATPerformance:
    def test_isolation_throughput(self):
        n = MATNeuron()
        N = 100_000
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(30.0)
        elapsed = time.perf_counter() - t0
        rate = N / elapsed
        assert_throughput_guard(
            label="MAT isolation",
            observed_per_second=rate,
            strict_minimum_per_second=100_000.0,
            smoke_minimum_per_second=25_000.0,
        )

    def test_network_throughput(self):
        pop = Population(MATNeuron, n=20, label="bench")
        drive = PoissonInput(n=20, rate_hz=500.0, weight=30.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        t0 = time.perf_counter()
        net.run(duration=0.5, dt=0.001, backend="python")
        elapsed = time.perf_counter() - t0
        neuron_steps = 20 * 500
        rate = neuron_steps / elapsed
        assert rate > 2_000, f"network: {rate:.0f} neuron-steps/s"
