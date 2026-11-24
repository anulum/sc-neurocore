# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestHayPerformance from former test_model_hay_l5.py

"""Focused suite: TestHayPerformance from former test_model_hay_l5.py."""

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
from sc_neurocore.neurons.models.hay_l5 import (
    HayL5PyramidalNeuron,
)
from tests.performance_guard import assert_load_tolerant_throughput


class TestHayPerformance:
    """Throughput of the HayL5PyramidalNeuron."""

    def test_isolation_throughput(self) -> None:
        """The isolated model meets its throughput floor."""
        n = HayL5PyramidalNeuron()
        N = 2000
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(10.0)
        elapsed = time.perf_counter() - t0
        rate = N / elapsed
        # 4 sub-steps × 3 compartments × multiple currents
        assert_load_tolerant_throughput(
            label="Hay L5 isolation", observed_per_second=rate, strict_minimum_per_second=500.0
        )

    def test_network_throughput(self) -> None:
        """The network meets its throughput floor."""
        pop = Population(HayL5PyramidalNeuron, n=10, label="bench")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=10.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        t0 = time.perf_counter()
        net.run(duration=0.2, dt=0.001, backend="python")
        elapsed = time.perf_counter() - t0
        neuron_steps = 10 * 200
        rate = neuron_steps / elapsed
        assert_load_tolerant_throughput(
            label="Hay L5 network", observed_per_second=rate, strict_minimum_per_second=100.0
        )
