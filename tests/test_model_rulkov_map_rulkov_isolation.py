# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestRulkovIsolation from former test_model_rulkov_map.py

"""Focused suite: TestRulkovIsolation from former test_model_rulkov_map.py."""

from __future__ import annotations

import numpy as np

from sc_neurocore.neurons.models.rulkov_map import RulkovMapNeuron


class TestRulkovIsolation:
    def test_construction_defaults(self) -> None:
        n = RulkovMapNeuron()
        assert n.x == -1.0
        assert n.y == -3.0
        assert n.alpha == 4.0
        assert n.sigma == -1.6
        assert n.mu == 0.001

    def test_step_returns_binary(self) -> None:
        assert RulkovMapNeuron().step(0.0) in (0, 1)

    def test_state_evolves(self) -> None:
        n = RulkovMapNeuron()
        x0, y0 = n.x, n.y
        n.step(0.5)
        assert n.x != x0 or n.y != y0

    def test_state_finite_long_run(self) -> None:
        n = RulkovMapNeuron()
        for _ in range(50000):
            n.step(0.5)
        assert np.isfinite(n.x) and np.isfinite(n.y)

    def test_reset(self) -> None:
        n = RulkovMapNeuron()
        for _ in range(100):
            n.step(1.0)
        n.reset()
        assert n.x == -1.0 and n.y == -3.0
