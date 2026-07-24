# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestWendlingIsolation from former test_model_wendling.py

"""Focused suite: TestWendlingIsolation from former test_model_wendling.py."""

from __future__ import annotations

from tests.model_wendling_support import *  # noqa: F403


class TestWendlingIsolation:
    def test_defaults(self):
        n = WendlingNeuron()
        assert n.y0 == 0.0 and n.y1 == 0.0
        assert n.a_exc == 3.25 and n.b_fast == 22.0
        assert n.dt == 0.001

    def test_step_returns_float(self):
        """Returns EEG signal (float), not binary spike."""
        n = WendlingNeuron()
        result = n.step(220.0)
        assert isinstance(result, (float, np.floating))

    def test_eight_state_variables_evolve(self):
        n = WendlingNeuron()
        initial = [n.y0, n.y1, n.y2, n.y3, n.y5, n.y6, n.y7, n.y8]
        for _ in range(1000):
            n.step(220.0)
        final = [n.y0, n.y1, n.y2, n.y3, n.y5, n.y6, n.y7, n.y8]
        for i, (v0, v1) in enumerate(zip(initial, final)):
            assert v0 != v1, f"State {i} didn't evolve"

    def test_state_finite(self):
        n = WendlingNeuron()
        for _ in range(100000):
            n.step(220.0)
        for attr in ["y0", "y1", "y2", "y3", "y5", "y6", "y7", "y8"]:
            assert np.isfinite(getattr(n, attr)), f"{attr} not finite"

    def test_reset(self):
        n = WendlingNeuron()
        for _ in range(1000):
            n.step(220.0)
        n.y4 = 1.0
        n.y9 = -1.0
        n.reset()
        assert n.y0 == 0.0 and n.y1 == 0.0 and n.y2 == 0.0 and n.y3 == 0.0
        assert n.y4 == 0.0 and n.y9 == 0.0
