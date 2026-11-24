# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPerformance from former test_model_gamma_renewal.py

"""Focused suite: TestPerformance from former test_model_gamma_renewal.py."""

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
from sc_neurocore.neurons.models.gamma_renewal import (
    GammaRenewalNeuron,
)
from tests.performance_guard import assert_throughput_guard


class TestPerformance:
    """Throughput of the GammaRenewalNeuron."""

    def test_isolation_throughput(self) -> None:
        """The isolated model meets its throughput floor."""
        n = GammaRenewalNeuron()
        N = 20000
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(100.0)
        elapsed = time.perf_counter() - t0
        assert_throughput_guard(
            label="Gamma-renewal isolation",
            observed_per_second=N / elapsed,
            strict_minimum_per_second=20_000.0,
            smoke_minimum_per_second=100.0,
        )

    def test_network_throughput(self) -> None:
        """The network meets its throughput floor."""
        pop = Population(GammaRenewalNeuron, n=50, label="bench")
        drive = PoissonInput(n=50, rate_hz=500.0, weight=100.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        t0 = time.perf_counter()
        net.run(duration=0.5, dt=0.001, backend="python")
        elapsed = time.perf_counter() - t0
        assert_throughput_guard(
            label="Gamma-renewal network",
            observed_per_second=50 * 500 / elapsed,
            strict_minimum_per_second=5_000.0,
            smoke_minimum_per_second=100.0,
        )
