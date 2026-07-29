# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDendrifyPerformance from former test_model_dendrify.py

"""Focused suite: TestDendrifyPerformance from former test_model_dendrify.py."""

from __future__ import annotations

from tests.model_dendrify_support import *  # noqa: F403
from tests.performance_guard import assert_load_tolerant_throughput


class TestDendrifyPerformance:
    def test_isolation_throughput(self):
        n = DendrifyNeuron()
        N = 20000
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(50.0)
        elapsed = time.perf_counter() - t0
        assert_load_tolerant_throughput(
            label="Dendrify isolation",
            observed_per_second=N / elapsed,
            strict_minimum_per_second=20_000.0,
        )
