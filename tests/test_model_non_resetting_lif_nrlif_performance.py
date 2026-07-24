# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestNRLIFPerformance from former test_model_non_resetting_lif.py

"""Focused suite: TestNRLIFPerformance from former test_model_non_resetting_lif.py."""

from __future__ import annotations

from tests.model_non_resetting_lif_support import *  # noqa: F403


class TestNRLIFPerformance:
    def test_isolation_runtime_regression_sentinel(self):
        """Bound pathological slowdowns without making CI throughput claims."""
        n = NonResettingLIFNeuron()
        N = 200_000
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(20.0)
        elapsed = time.perf_counter() - t0
        assert np.isfinite(n.v) and np.isfinite(n.theta)
        assert elapsed < 10.0

    def test_network_runtime_regression_sentinel(self):
        """Bound pathological network slowdowns without throughput claims."""
        pop = Population(NonResettingLIFNeuron, n=20, label="bench")
        drive = PoissonInput(n=20, rate_hz=500.0, weight=20.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        t0 = time.perf_counter()
        net.run(duration=0.5, dt=0.001, backend="python")
        elapsed = time.perf_counter() - t0
        assert elapsed < 10.0
        assert mon.count >= 0
