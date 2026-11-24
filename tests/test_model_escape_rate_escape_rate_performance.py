# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestEscapeRatePerformance from former test_model_escape_rate.py

"""Focused suite: TestEscapeRatePerformance from former test_model_escape_rate.py."""

from __future__ import annotations

import time
from sc_neurocore.network.monitor import (
    SpikeMonitor,
)
from sc_neurocore.network.network import (
    Network,
)
from sc_neurocore.network.population import (
    Population,
)
from sc_neurocore.network.stimulus import (
    PoissonInput,
)
from sc_neurocore.neurons.models.escape_rate import (
    EscapeRateNeuron,
)
from tests.performance_guard import assert_load_tolerant_throughput


class TestEscapeRatePerformance:
    """Throughput of the EscapeRateNeuron."""

    def test_isolation_throughput(self) -> None:
        """The isolated model meets its throughput floor."""
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
        assert_load_tolerant_throughput(
            label="Escape-rate isolation",
            observed_per_second=N / elapsed,
            strict_minimum_per_second=10_000.0,
        )

    def test_network_throughput(self) -> None:
        """The network meets its throughput floor."""
        pop = Population(EscapeRateNeuron, n=50, label="bench")
        drive = PoissonInput(n=50, rate_hz=500.0, weight=50.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        t0 = time.perf_counter()
        net.run(duration=0.5, dt=0.001, backend="python")
        elapsed = time.perf_counter() - t0
        assert_load_tolerant_throughput(
            label="Escape-rate network",
            observed_per_second=50 * 500 / elapsed,
            strict_minimum_per_second=5_000.0,
        )
