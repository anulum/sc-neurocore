# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestThetaIsolation from former test_model_theta.py

"""Focused suite: TestThetaIsolation from former test_model_theta.py."""

from __future__ import annotations

from tests.model_theta_support import *  # noqa: F403

class TestThetaIsolation:
    def test_construction_defaults(self) -> None:
        n = ThetaNeuron()
        assert n.theta == 0.0
        assert n.dt == 0.01

    def test_step_returns_binary(self) -> None:
        assert ThetaNeuron().step(0.0) in (0, 1)

    def test_theta_evolves(self) -> None:
        n = ThetaNeuron()
        n.step(1.0)
        assert n.theta != 0.0

    def test_theta_wrapped_to_minus_pi_pi(self) -> None:
        """theta is wrapped to [-π, π] after each step."""
        n = ThetaNeuron()
        for _ in range(10000):
            n.step(5.0)
        assert -np.pi <= n.theta <= np.pi

    def test_state_finite_long_run(self) -> None:
        n = ThetaNeuron()
        for _ in range(100000):
            n.step(2.0)
        assert np.isfinite(n.theta)

    def test_reset(self) -> None:
        n = ThetaNeuron()
        for _ in range(100):
            n.step(2.0)
        n.reset()
        assert n.theta == 0.0
