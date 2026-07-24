# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFHRIsolation from former test_model_fitzhugh_rinzel.py

"""Focused suite: TestFHRIsolation from former test_model_fitzhugh_rinzel.py."""

from __future__ import annotations

from tests.model_fitzhugh_rinzel_support import *  # noqa: F403


class TestFHRIsolation:
    def test_defaults(self):
        n = FitzHughRinzelNeuron()
        assert n.v == -1.0 and n.w == -0.5 and n.y == 0.0
        assert n.delta == 0.08 and n.mu == 0.0001
        assert n.b == 0.8 and n.d == 1.0

    def test_step_returns_binary(self):
        assert FitzHughRinzelNeuron().step(0.0) in (0, 1)

    def test_three_variables_evolve(self):
        n = FitzHughRinzelNeuron()
        initial = (n.v, n.w, n.y)
        for _ in range(1000):
            n.step(0.5)
        for name, v0, v1 in zip(["v", "w", "y"], initial, (n.v, n.w, n.y), strict=True):
            assert v0 != v1, f"{name} did not evolve"

    def test_state_finite(self):
        n = FitzHughRinzelNeuron()
        for _ in range(100_000):
            n.step(0.5)
        assert np.isfinite(n.v) and np.isfinite(n.w) and np.isfinite(n.y)

    def test_reset(self):
        n = FitzHughRinzelNeuron()
        for _ in range(500):
            n.step(0.5)
        n.reset()
        assert n.v == -1.0 and n.w == -0.5 and n.y == 0.0
