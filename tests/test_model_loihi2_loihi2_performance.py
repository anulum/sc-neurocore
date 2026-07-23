# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLoihi2Performance from former test_model_loihi2.py

"""Focused suite: TestLoihi2Performance from former test_model_loihi2.py."""

from __future__ import annotations

from tests.model_loihi2_support import *  # noqa: F403

class TestLoihi2Performance:
    def test_isolation_throughput(self):
        n = Loihi2Neuron()
        N = 500_000
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(200)
        elapsed = time.perf_counter() - t0
        assert N / elapsed > 200_000, f"isolation: {N / elapsed:.0f} steps/s"

    def test_network_throughput(self):
        pop = Population(Loihi2Neuron, n=20, label="bench")
        drive = PoissonInput(n=20, rate_hz=500.0, weight=200.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        t0 = time.perf_counter()
        net.run(duration=0.5, dt=0.001, backend="python")
        elapsed = time.perf_counter() - t0
        assert 20 * 500 / elapsed > 2_000
