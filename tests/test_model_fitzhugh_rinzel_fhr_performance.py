# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFHRPerformance from former test_model_fitzhugh_rinzel.py

"""Focused suite: TestFHRPerformance from former test_model_fitzhugh_rinzel.py."""

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
from sc_neurocore.neurons.models.fitzhugh_rinzel import (
    FitzHughRinzelNeuron,
)
from tests.performance_guard import assert_load_tolerant_throughput


class TestFHRPerformance:
    """Throughput of the FitzHughRinzelNeuron."""

    def test_isolation_throughput(self) -> None:
        """The isolated model meets its throughput floor."""
        samples = []
        for _ in range(3):
            n = FitzHughRinzelNeuron()
            steps = 50_000
            t0 = time.perf_counter()
            for _ in range(steps):
                n.step(0.5)
            samples.append(time.perf_counter() - t0)

        best_steps_per_second = steps / min(samples)
        assert_load_tolerant_throughput(
            label="FitzHugh-Rinzel isolation",
            observed_per_second=best_steps_per_second,
            strict_minimum_per_second=10_000.0,
        )

    def test_network_throughput(self) -> None:
        """The network meets its throughput floor."""
        pop = Population(FitzHughRinzelNeuron, n=50, label="bench")
        drive = PoissonInput(n=50, rate_hz=500.0, weight=0.8, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        t0 = time.perf_counter()
        net.run(duration=0.5, dt=0.001, backend="python")
        elapsed = time.perf_counter() - t0
        assert_load_tolerant_throughput(
            label="FitzHugh-Rinzel network",
            observed_per_second=50 * 500 / elapsed,
            strict_minimum_per_second=3_000.0,
        )
