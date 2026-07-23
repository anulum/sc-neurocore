# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestAIOptimizedPerformance from former test_model_ai_optimized.py

"""Focused suite: TestAIOptimizedPerformance from former test_model_ai_optimized.py."""

from __future__ import annotations

from tests.model_ai_optimized_support import *  # noqa: F403

class TestAIOptimizedPerformance:
    @pytest.mark.parametrize(
        "cls,min_perf",
        [
            (MetaPlasticNeuron, 100000),
            (DifferentiableSurrogateNeuron, 100000),
            (AttentionGatedNeuron, 100000),
            (MultiTimescaleNeuron, 50000),
            (ContinuousAttractorNeuron, 1000),
        ],
        ids=lambda c: c.__name__ if isinstance(c, type) else str(c),
    )
    def test_throughput(self, cls: type, min_perf: int):
        n = cls()
        N = min(min_perf, 10000)
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(2.0)
        elapsed = time.perf_counter() - t0
        strict_minimum = float(min_perf) * 0.5
        assert_throughput_guard(
            label=f"{cls.__name__} isolation",
            observed_per_second=N / elapsed,
            strict_minimum_per_second=strict_minimum,
            smoke_minimum_per_second=min(500.0, max(25.0, strict_minimum * 0.01)),
        )
