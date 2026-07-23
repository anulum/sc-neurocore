# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestAIOptimizedAnalysis from former test_model_ai_optimized.py

"""Focused suite: TestAIOptimizedAnalysis from former test_model_ai_optimized.py."""

from __future__ import annotations

from tests.model_ai_optimized_support import *  # noqa: F403

class TestAIOptimizedAnalysis:
    def test_spike_count_multitimescale(self):
        n = MultiTimescaleNeuron()
        train = np.array([float(n.step(2.0)) for _ in range(5000)])
        assert spike_count(train) >= 50

    def test_spike_count_self_referential(self):
        n = SelfReferentialNeuron()
        train = np.array([float(n.step(2.0)) for _ in range(5000)])
        assert spike_count(train) >= 10
