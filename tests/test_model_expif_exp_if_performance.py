# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestExpIFPerformance from former test_model_expif.py

"""Focused suite: TestExpIFPerformance from former test_model_expif.py."""

from __future__ import annotations

from tests.model_expif_support import *  # noqa: F403
from tests.performance_guard import assert_load_tolerant_throughput


class TestExpIFPerformance:
    def test_isolation_throughput(self) -> None:
        neuron = ExpIFNeuron()
        steps = 50_000
        started = time.perf_counter()
        for _ in range(steps):
            neuron.step(20.0)
        rate = steps / (time.perf_counter() - started)
        minimum = 10_000 if os.getenv("CI") else 12_000
        assert_load_tolerant_throughput(
            label="ExpIF isolation",
            observed_per_second=rate,
            strict_minimum_per_second=float(minimum),
        )

    def test_network_throughput(self) -> None:
        population = Population(ExpIFNeuron, n=50, label="bench")
        drive = PoissonInput(n=50, rate_hz=500.0, weight=50.0, dt=0.001, seed=42)
        monitor = SpikeMonitor(population)
        network = Network(population, drive, monitor)
        started = time.perf_counter()
        network.run(duration=0.5, dt=0.001, backend="python")
        elapsed = time.perf_counter() - started
        assert_load_tolerant_throughput(
            label="ExpIF network",
            observed_per_second=50 * 500 / elapsed,
            strict_minimum_per_second=5_000.0,
        )
