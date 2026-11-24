# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestArcanePerformance from former test_model_arcane_neuron.py

"""Focused suite: TestArcanePerformance from former test_model_arcane_neuron.py."""

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
from sc_neurocore.neurons.models.arcane_neuron import (
    ArcaneNeuron,
)
from tests.performance_guard import assert_load_tolerant_throughput


class TestArcanePerformance:
    """Throughput of the Arcane neuron."""

    def test_isolation_throughput(self) -> None:
        """The isolated model meets its throughput floor."""
        n = ArcaneNeuron()
        N = 10000
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(2.0)
        elapsed = time.perf_counter() - t0
        assert_load_tolerant_throughput(
            label="Arcane isolation",
            observed_per_second=N / elapsed,
            strict_minimum_per_second=1_000.0,
        )

    def test_network_throughput(self) -> None:
        """The network meets its throughput floor."""
        pop = Population(ArcaneNeuron, n=10, label="bench")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=3.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        t0 = time.perf_counter()
        net.run(duration=0.5, dt=0.001, backend="python")
        elapsed = time.perf_counter() - t0
        neuron_steps = 10 * 500
        assert_load_tolerant_throughput(
            label="Arcane network",
            observed_per_second=neuron_steps / elapsed,
            strict_minimum_per_second=500.0,
        )
