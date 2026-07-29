# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestAstrocyteAdapterPerformance from former test_model_astrocyte_adapter.py

"""Focused suite: TestAstrocyteAdapterPerformance from former test_model_astrocyte_adapter.py."""

from __future__ import annotations

from tests.model_astrocyte_adapter_support import *  # noqa: F403
from tests.performance_guard import assert_load_tolerant_throughput


class TestAstrocyteAdapterPerformance:
    """Smoke tests for local adapter throughput budgets."""

    def test_isolation_throughput(self) -> None:
        """Single-adapter stepping stays above the local smoke threshold."""
        n = AstrocyteNeuron()
        N = 20000
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(0.5)
        elapsed = time.perf_counter() - t0
        assert_load_tolerant_throughput(
            label="Astrocyte adapter isolation",
            observed_per_second=N / elapsed,
            strict_minimum_per_second=20_000.0,
        )

    def test_network_throughput(self) -> None:
        """Network execution stays above the local smoke threshold."""
        pop = Population(AstrocyteNeuron, n=20, label="bench")
        drive = PoissonInput(n=20, rate_hz=200.0, weight=0.5, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        t0 = time.perf_counter()
        net.run(duration=0.5, dt=0.001, backend="python")
        elapsed = time.perf_counter() - t0
        neuron_steps = 20 * 500
        assert_load_tolerant_throughput(
            label="Astrocyte adapter network",
            observed_per_second=neuron_steps / elapsed,
            strict_minimum_per_second=2_000.0,
        )
