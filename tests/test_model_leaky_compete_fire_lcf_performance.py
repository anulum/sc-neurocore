# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLCFPerformance from former test_model_leaky_compete_fire.py

"""Focused suite: TestLCFPerformance from former test_model_leaky_compete_fire.py."""

from __future__ import annotations

import time
from sc_neurocore.neurons.models.leaky_compete_fire import (
    LeakyCompeteFireNeuron,
)
from tests.performance_guard import assert_load_tolerant_throughput


class TestLCFPerformance:
    """Throughput of the LeakyCompeteFireNeuron."""

    def test_isolation_throughput(self) -> None:
        """The isolated model meets its throughput floor."""
        n = LeakyCompeteFireNeuron()
        N = 100_000
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(5.0)
        elapsed = time.perf_counter() - t0
        rate = N / elapsed
        assert_load_tolerant_throughput(
            label="Leaky compete-fire isolation",
            observed_per_second=rate,
            strict_minimum_per_second=10_000.0,
        )
