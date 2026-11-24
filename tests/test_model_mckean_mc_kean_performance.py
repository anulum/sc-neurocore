# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestMcKeanPerformance from former test_model_mckean.py

"""Focused suite: TestMcKeanPerformance from former test_model_mckean.py."""

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
from sc_neurocore.neurons.models.sc_triangular_mckean import (
    SCTriangularMcKeanNeuron as McKeanNeuron,
)
from tests.performance_guard import assert_load_tolerant_throughput


class TestMcKeanPerformance:
    """Throughput of the McKeanNeuron."""

    def test_isolation_throughput(self) -> None:
        """The isolated model meets its throughput floor."""
        n = McKeanNeuron()
        steps = 200_000
        t0 = time.perf_counter()
        for _ in range(steps):
            n.step(0.5)
        elapsed = time.perf_counter() - t0
        rate = steps / elapsed
        assert_load_tolerant_throughput(
            label="McKean isolation", observed_per_second=rate, strict_minimum_per_second=10_000.0
        )

    def test_network_throughput(self) -> None:
        """The network meets its throughput floor."""
        pop = Population(McKeanNeuron, n=20, label="bench")
        drive = PoissonInput(n=20, rate_hz=500.0, weight=0.5, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        t0 = time.perf_counter()
        net.run(duration=0.5, dt=0.001, backend="python")
        elapsed = time.perf_counter() - t0
        neuron_steps = 20 * 500
        rate = neuron_steps / elapsed
        assert_load_tolerant_throughput(
            label="McKean network", observed_per_second=rate, strict_minimum_per_second=2_000.0
        )
