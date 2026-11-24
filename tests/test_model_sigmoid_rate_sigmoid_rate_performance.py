# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSigmoidRatePerformance from former test_model_sigmoid_rate.py

"""Focused suite: TestSigmoidRatePerformance from former test_model_sigmoid_rate.py."""

from __future__ import annotations

import time
from sc_neurocore.neurons.models.sigmoid_rate import (
    SigmoidRateNeuron,
)
from tests.performance_guard import assert_load_tolerant_throughput


class TestSigmoidRatePerformance:
    """Throughput of the SigmoidRateNeuron."""

    def test_isolation_throughput(self) -> None:
        """The isolated model meets its throughput floor."""
        n = SigmoidRateNeuron()
        N = 100000
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(3.0)
        elapsed = time.perf_counter() - t0
        assert_load_tolerant_throughput(
            label="Sigmoid-rate isolation",
            observed_per_second=N / elapsed,
            strict_minimum_per_second=50_000.0,
        )
