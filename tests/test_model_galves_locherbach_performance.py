# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPerformance from former test_model_galves_locherbach.py

"""Focused suite: TestPerformance from former test_model_galves_locherbach.py."""

from __future__ import annotations

from tests.model_galves_locherbach_support import *  # noqa: F403

class TestPerformance:
    def test_isolation_throughput(self):
        n = GalvesLocherbachNeuron()
        N = 50000
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(10.0)
        elapsed = time.perf_counter() - t0
        assert N / elapsed > 50000

    def test_network_throughput(self):
        pop = Population(GalvesLocherbachNeuron, n=50, label="bench")
        drive = PoissonInput(n=50, rate_hz=500.0, weight=10.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        t0 = time.perf_counter()
        net.run(duration=0.5, dt=0.001, backend="python")
        elapsed = time.perf_counter() - t0
        assert 50 * 500 / elapsed > 5000
