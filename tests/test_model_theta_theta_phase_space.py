# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestThetaPhaseSpace from former test_model_theta.py

"""Focused suite: TestThetaPhaseSpace from former test_model_theta.py."""

from __future__ import annotations

from tests.model_theta_support import *  # noqa: F403

class TestThetaPhaseSpace:
    """Phase dynamics on the unit circle."""

    def test_theta_traverses_full_circle(self) -> None:
        """At I>0, theta should cycle through [-π, π]."""
        n = ThetaNeuron()
        thetas = set()
        for _ in range(10000):
            n.step(1.0)
            thetas.add(round(n.theta, 1))
        # Should visit many distinct theta values
        assert len(thetas) > 20

    def test_spike_at_pi(self) -> None:
        """Spike is detected when the exact flow crosses π from below."""
        n = ThetaNeuron()
        for _ in range(50000):
            if n.step(1.0) == 1:
                # After spike, theta is wrapped. Just verify spike occurred
                return
        pytest.fail("No spike in 50k steps at I=1.0")

    def test_dynamics_equation(self) -> None:
        """Verify the tangent-half-angle exact constant-current flow."""
        n = ThetaNeuron(theta=1.0)
        expected, spiked = _exact_theta_candidate(n.theta, 2.0, n.dt)
        result = n.step(2.0)
        assert result == int(spiked)
        assert abs(n.theta - expected) < 1e-12

    def test_exact_positive_flow_separates_from_forward_euler(self) -> None:
        n = ThetaNeuron(theta=1.0, dt=0.2)
        current = 2.0
        euler = _wrap_phase(
            n.theta + ((1.0 - math.cos(n.theta)) + (1.0 + math.cos(n.theta)) * current) * n.dt
        )
        expected, spiked = _exact_theta_candidate(n.theta, current, n.dt)
        result = n.step(current)
        assert result == int(spiked)
        assert abs(n.theta - expected) < 1e-12
        assert abs(n.theta - euler) > 1e-4

    def test_exact_flow_reports_within_step_crossing(self) -> None:
        n = ThetaNeuron(theta=2.5, dt=1.0)
        expected, spiked = _exact_theta_candidate(n.theta, 1.0, n.dt)
        assert spiked
        assert n.step(1.0) == 1
        assert abs(n.theta - expected) < 1e-12

    def test_negative_current_stable_fixed_point_is_preserved(self) -> None:
        n = ThetaNeuron(theta=-math.pi / 2.0, dt=100.0)
        assert n.step(-1.0) == 0
        assert abs(n.theta + math.pi / 2.0) < 1e-12
