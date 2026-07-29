# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestWendlingPerformance from former test_model_wendling.py

"""Focused suite: TestWendlingPerformance from former test_model_wendling.py."""

from __future__ import annotations

from tests.model_wendling_support import *  # noqa: F403
from tests.performance_guard import assert_load_tolerant_throughput


class TestWendlingPerformance:
    def test_isolation_throughput(self):
        n = WendlingNeuron()
        N = 20000
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(220.0)
        elapsed = time.perf_counter() - t0
        assert_load_tolerant_throughput(
            label="Wendling isolation",
            observed_per_second=N / elapsed,
            strict_minimum_per_second=10_000.0,
        )
