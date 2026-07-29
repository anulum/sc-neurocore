# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestGFSPerformance from former test_model_golomb_fs.py

"""Focused suite: TestGFSPerformance from former test_model_golomb_fs.py."""

from __future__ import annotations

from tests.model_golomb_fs_support import *  # noqa: F403
from tests.performance_guard import assert_load_tolerant_throughput


class TestGFSPerformance:
    def test_isolation_throughput(self):
        n = GolombFSNeuron()
        N = 2000
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(5.0)
        elapsed = time.perf_counter() - t0
        rate = N / elapsed
        # 10 sub-steps × HH
        assert_load_tolerant_throughput(
            label="Golomb FS isolation", observed_per_second=rate, strict_minimum_per_second=500.0
        )

    def test_network_throughput(self):
        pop = Population(GolombFSNeuron, n=10, label="bench")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=5.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        t0 = time.perf_counter()
        net.run(duration=0.2, dt=0.001, backend="python")
        elapsed = time.perf_counter() - t0
        neuron_steps = 10 * 200
        rate = neuron_steps / elapsed
        assert_load_tolerant_throughput(
            label="Golomb FS network", observed_per_second=rate, strict_minimum_per_second=100.0
        )
