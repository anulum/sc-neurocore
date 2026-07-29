# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestNLIFPerformance from former test_model_nlif.py

"""Focused suite: TestNLIFPerformance from former test_model_nlif.py."""

from __future__ import annotations

from tests.model_nlif_support import *  # noqa: F403
from tests.performance_guard import assert_load_tolerant_throughput


def _isolation_min_rate(*, ci_rate: int, local_rate: int) -> int:
    """Floor under CI / coverage tracing vs uninstrumented local runs."""
    if os.environ.get("CI"):
        return ci_rate
    try:
        from coverage import Coverage

        if Coverage.current() is not None:
            return ci_rate
    except Exception:
        pass
    return local_rate


class TestNLIFPerformance:
    def test_isolation_throughput(self):
        n = NonlinearLIFNeuron()
        N = 200_000
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(20.0)
        elapsed = time.perf_counter() - t0
        rate = N / elapsed
        min_rate = _isolation_min_rate(ci_rate=50_000, local_rate=100_000)
        assert_load_tolerant_throughput(
            label="NLIF isolation",
            observed_per_second=rate,
            strict_minimum_per_second=float(min_rate),
        )

    def test_network_throughput(self):
        pop = Population(NonlinearLIFNeuron, n=20, label="bench")
        drive = PoissonInput(n=20, rate_hz=500.0, weight=20.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        t0 = time.perf_counter()
        net.run(duration=0.5, dt=0.001, backend="python")
        elapsed = time.perf_counter() - t0
        neuron_steps = 20 * 500
        rate = neuron_steps / elapsed
        assert_load_tolerant_throughput(
            label="NLIF network", observed_per_second=rate, strict_minimum_per_second=2_000.0
        )
