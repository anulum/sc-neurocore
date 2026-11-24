# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBertramPerformance from former test_model_bertram_phantom.py

"""Focused suite: TestBertramPerformance from former test_model_bertram_phantom.py."""

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
from sc_neurocore.neurons.models.sc_three_state_phantom import (
    SCThreeStatePhantomBurster as BertramPhantomBurster,
)
from tests.performance_guard import assert_load_tolerant_throughput


class TestBertramPerformance:
    """Throughput of the BertramPhantomBurster."""

    def test_isolation_runtime_regression_sentinel(self) -> None:
        """Bound pathological slowdowns without making CI throughput claims."""
        n = BertramPhantomBurster()
        N = 50_000
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(200.0)
        elapsed = time.perf_counter() - t0
        # 4 RK4 stages: 16 Boltzmann evaluations + 20 currents per step.
        # Production throughput belongs in isolated benchmark artifacts, not CI.
        assert_load_tolerant_throughput(
            label="Bertram isolation run",
            observed_per_second=1.0 / elapsed,
            strict_minimum_per_second=1.0 / 15.0,
        )
        assert np.isfinite(n.v) and np.isfinite(n.s1) and np.isfinite(n.s2)

    def test_network_runtime_regression_sentinel(self) -> None:
        """Bound pathological network slowdowns without throughput claims."""
        pop = Population(BertramPhantomBurster, n=20, label="bench")
        drive = PoissonInput(n=20, rate_hz=1000.0, weight=200.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        t0 = time.perf_counter()
        net.run(duration=0.5, dt=0.001, backend="python")
        elapsed = time.perf_counter() - t0
        assert_load_tolerant_throughput(
            label="Bertram network run",
            observed_per_second=1.0 / elapsed,
            strict_minimum_per_second=1.0 / 15.0,
        )
        assert mon.count >= 0
