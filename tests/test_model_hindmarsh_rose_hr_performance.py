# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestHRPerformance from former test_model_hindmarsh_rose.py

"""Focused suite: TestHRPerformance from former test_model_hindmarsh_rose.py."""

from __future__ import annotations

from tests.model_hindmarsh_rose_support import *  # noqa: F403


class TestHRPerformance:
    def test_isolation_throughput(self):
        n = HindmarshRoseNeuron()
        N = 200_000
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(5.0)
        elapsed = time.perf_counter() - t0
        rate = N / elapsed
        min_rate = 60_000 if os.getenv("CI") else 200_000
        assert np.isfinite(n.x) and np.isfinite(n.y) and np.isfinite(n.z)
        assert rate > min_rate, f"isolation: {rate:.0f} steps/s, minimum={min_rate}"

    def test_network_throughput(self):
        pop = Population(HindmarshRoseNeuron, n=20, label="bench")
        drive = PoissonInput(n=20, rate_hz=500.0, weight=5.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        t0 = time.perf_counter()
        net.run(duration=0.5, dt=0.001, backend="python")
        elapsed = time.perf_counter() - t0
        neuron_steps = 20 * 500
        rate = neuron_steps / elapsed
        assert rate > 2_000, f"network: {rate:.0f} neuron-steps/s"
