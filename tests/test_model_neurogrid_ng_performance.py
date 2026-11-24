# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestNGPerformance from former test_model_neurogrid.py

"""Focused suite: TestNGPerformance from former test_model_neurogrid.py."""

from __future__ import annotations

import numpy as np
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
from sc_neurocore.neurons.models.neurogrid import (
    NeuroGridNeuron,
)
from tests.performance_guard import (
    assert_throughput_guard,
)


class TestNGPerformance:
    """Throughput of the NeuroGridNeuron."""

    def test_isolation_throughput(self) -> None:
        """The isolated model meets its throughput floor."""
        n = NeuroGridNeuron()
        N = 100_000
        t0 = time.process_time()
        for _ in range(N):
            n.step(100.0)
        elapsed = time.process_time() - t0
        rate = N / elapsed
        assert np.isfinite(n.v_s) and np.isfinite(n.v_d)
        assert_throughput_guard(
            label="NeuroGrid isolation",
            observed_per_second=rate,
            strict_minimum_per_second=20_000.0,
            smoke_minimum_per_second=100.0,
        )

    def test_network_throughput(self) -> None:
        """The network meets its throughput floor."""
        pop = Population(NeuroGridNeuron, n=20, label="bench")
        drive = PoissonInput(n=20, rate_hz=500.0, weight=100.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        t0 = time.perf_counter()
        net.run(duration=0.5, dt=0.001, backend="python")
        elapsed = time.perf_counter() - t0
        neuron_steps = 20 * 500
        rate = neuron_steps / elapsed
        assert_throughput_guard(
            label="NeuroGrid network",
            observed_per_second=rate,
            strict_minimum_per_second=2_000.0,
            smoke_minimum_per_second=100.0,
        )
