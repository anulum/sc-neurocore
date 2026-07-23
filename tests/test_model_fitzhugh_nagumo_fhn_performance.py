# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFHNPerformance from former test_model_fitzhugh_nagumo.py

"""Focused suite: TestFHNPerformance from former test_model_fitzhugh_nagumo.py."""

from __future__ import annotations

from tests.model_fitzhugh_nagumo_support import *  # noqa: F403

class TestFHNPerformance:
    def test_isolation_throughput(self):
        n = FitzHughNagumoNeuron()
        N = 100000
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(0.8)
        elapsed = time.perf_counter() - t0
        throughput = N / elapsed
        minimum_throughput = 10000 if os.environ.get("CI") else 20000
        assert np.isfinite(n.v) and np.isfinite(n.w)
        assert_throughput_guard(
            label="FHN isolation",
            observed_per_second=throughput,
            strict_minimum_per_second=float(minimum_throughput),
            smoke_minimum_per_second=100.0,
        )

    def test_network_throughput(self):
        pop = Population(FitzHughNagumoNeuron, n=50, label="bench")
        drive = PoissonInput(n=50, rate_hz=500.0, weight=0.8, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        t0 = time.perf_counter()
        net.run(duration=0.5, dt=0.001, backend="python")
        elapsed = time.perf_counter() - t0
        assert_throughput_guard(
            label="FHN network",
            observed_per_second=50 * 500 / elapsed,
            strict_minimum_per_second=5000.0,
            smoke_minimum_per_second=100.0,
        )
