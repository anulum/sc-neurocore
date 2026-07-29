# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLoihiCUBAPerformance from former test_model_loihi_cuba.py

"""Focused suite: TestLoihiCUBAPerformance from former test_model_loihi_cuba.py."""

from __future__ import annotations

from tests.model_loihi_cuba_support import *  # noqa: F403
from tests.performance_guard import assert_load_tolerant_throughput


class TestLoihiCUBAPerformance:
    def test_isolation_throughput(self):
        n = LoihiCUBANeuron()
        N = 500_000
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(200)
        elapsed = time.perf_counter() - t0
        assert_load_tolerant_throughput(
            label="Loihi CUBA isolation",
            observed_per_second=N / elapsed,
            strict_minimum_per_second=500_000.0,
        )

    def test_network_throughput(self):
        pop = Population(LoihiCUBANeuron, n=20, label="bench")
        drive = PoissonInput(n=20, rate_hz=500.0, weight=200.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        t0 = time.perf_counter()
        net.run(duration=0.5, dt=0.001, backend="python")
        elapsed = time.perf_counter() - t0
        assert_load_tolerant_throughput(
            label="Loihi CUBA network",
            observed_per_second=20 * 500 / elapsed,
            strict_minimum_per_second=2_000.0,
        )
