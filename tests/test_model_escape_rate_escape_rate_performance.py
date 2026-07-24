# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestEscapeRatePerformance from former test_model_escape_rate.py

"""Focused suite: TestEscapeRatePerformance from former test_model_escape_rate.py."""

from __future__ import annotations

from tests.model_escape_rate_support import *  # noqa: F403


class TestEscapeRatePerformance:
    def test_isolation_throughput(self):
        n = EscapeRateNeuron()
        for _ in range(1000):
            n.step(30.0)
        N = 50000
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(30.0)
        elapsed = time.perf_counter() - t0
        # Coarse regression smoke only: CI runs this concurrently under xdist;
        # controlled performance claims live in bench_model_escape_rate.py.
        assert N / elapsed > 10000

    def test_network_throughput(self):
        pop = Population(EscapeRateNeuron, n=50, label="bench")
        drive = PoissonInput(n=50, rate_hz=500.0, weight=50.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        t0 = time.perf_counter()
        net.run(duration=0.5, dt=0.001, backend="python")
        elapsed = time.perf_counter() - t0
        assert 50 * 500 / elapsed > 5000
