# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestAmariPerformance from former test_model_amari_field.py

"""Focused suite: TestAmariPerformance from former test_model_amari_field.py."""

from __future__ import annotations

from tests.model_amari_field_support import *  # noqa: F403

class TestAmariPerformance:
    def test_isolation_throughput(self):
        n = AmariNeuralField(n=64)
        I = np.ones(64) * 0.5
        N = 5000
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(I)
        elapsed = time.perf_counter() - t0
        assert N / elapsed > 5000

    def test_network_throughput(self):
        pop = Population(AmariNeuralField, n=3, label="bench")
        drive = PoissonInput(n=3, rate_hz=100.0, weight=1.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        t0 = time.perf_counter()
        net.run(duration=0.5, dt=0.001, backend="python")
        elapsed = time.perf_counter() - t0
        assert 3 * 500 / elapsed > 100
