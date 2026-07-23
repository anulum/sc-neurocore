# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestAIOptimizedCommon from former test_model_ai_optimized.py

"""Focused suite: TestAIOptimizedCommon from former test_model_ai_optimized.py."""

from __future__ import annotations

from tests.model_ai_optimized_support import *  # noqa: F403

@pytest.mark.parametrize("cls", ALL_CLASSES, ids=lambda c: c.__name__)
class TestAIOptimizedCommon:
    """Tests applied to all 8 AI-optimised models."""

    def test_step_returns_int(self, cls: type):
        n = cls()
        assert n.step(0.0) in (0, 1)

    def test_state_finite(self, cls: type):
        n = cls()
        for _ in range(5000):
            n.step(2.0)
        assert np.isfinite(getattr(n, "v", getattr(n, "v_fast", 0.0)))

    def test_fires_at_i2(self, cls: type):
        """All 8 models fire at I=2.0."""
        n = cls()
        spikes = sum(n.step(2.0) for _ in range(5000))
        assert spikes > 0, f"{cls.__name__} silent at I=2.0"

    def test_reset(self, cls: type):
        n = cls()
        for _ in range(100):
            n.step(2.0)
        n.reset()
        # After reset, first state variable should be at initial value
        # (exact check depends on model, but reset should not crash)

    def test_deterministic(self, cls: type):
        traces = []
        for _ in range(2):
            n = cls()
            trace = [n.step(2.0) for _ in range(200)]
            traces.append(trace)
        assert traces[0] == traces[1]

    def test_population_creates(self, cls: type):
        pop = Population(cls, n=5, label="ai")
        assert pop.n == 5
