# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLapicquePerformance from former test_model_lapicque.py

"""Focused suite: TestLapicquePerformance from former test_model_lapicque.py."""

from __future__ import annotations

from tests.model_lapicque_support import *  # noqa: F403
from tests.performance_guard import assert_load_tolerant_throughput


class TestLapicquePerformance:
    def test_isolation_throughput(self):
        n = LapicqueNeuron()
        N = 500_000
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(20.0)
        elapsed = time.perf_counter() - t0
        rate = N / elapsed
        min_rate = 100_000 if os.environ.get("CI") else 160_000
        assert np.isfinite(n.v)
        assert_load_tolerant_throughput(
            label="Lapicque isolation",
            observed_per_second=rate,
            strict_minimum_per_second=float(min_rate),
        )

    def test_network_throughput(self):
        pop = Population(LapicqueNeuron, n=20, label="bench")
        drive = PoissonInput(n=20, rate_hz=500.0, weight=20.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        t0 = time.perf_counter()
        net.run(duration=0.5, dt=0.001, backend="python")
        elapsed = time.perf_counter() - t0
        neuron_steps = 20 * 500
        rate = neuron_steps / elapsed
        assert_load_tolerant_throughput(
            label="Lapicque network", observed_per_second=rate, strict_minimum_per_second=5_000.0
        )
