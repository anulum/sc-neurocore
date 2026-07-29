# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSpiNNaker2Performance from former test_model_spinnaker2.py

"""Focused suite: TestSpiNNaker2Performance from former test_model_spinnaker2.py."""

from __future__ import annotations

from tests.model_spinnaker2_support import *  # noqa: F403
from tests.performance_guard import assert_load_tolerant_throughput


class TestSpiNNaker2Performance:
    def test_isolation_throughput(self):
        n = SpiNNaker2Neuron()
        N = 100000
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(500)
        elapsed = time.perf_counter() - t0
        assert_load_tolerant_throughput(
            label="SpiNNaker 2 isolation",
            observed_per_second=N / elapsed,
            strict_minimum_per_second=100_000.0,
        )
