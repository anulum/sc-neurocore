# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestThetaEdgeCases from former test_model_theta.py

"""Focused suite: TestThetaEdgeCases from former test_model_theta.py."""

from __future__ import annotations

from tests.model_theta_support import *  # noqa: F403

class TestThetaEdgeCases:
    def test_theta_wrapping_correct(self) -> None:
        """After large positive dtheta, theta stays in [-π, π]."""
        n = ThetaNeuron(theta=3.0, dt=0.5)
        n.step(10.0)  # large jump
        assert -np.pi <= n.theta <= np.pi

    def test_candidate_phase_is_validated_before_assignment(self) -> None:
        n = ThetaNeuron(theta=0.25, dt=1.0e308)
        before = n.theta
        with pytest.raises(ValueError, match="exact-flow candidate"):
            n.step(-1.0e308)
        assert n.theta == before

    def test_positive_flow_singularity_wraps_and_reports_crossing(self) -> None:
        n = ThetaNeuron(theta=0.0, dt=math.pi / 2.0)
        assert n.step(1.0) == 1
        assert n.theta == -math.pi

    def test_zero_current_singularity_wraps_and_reports_crossing(self) -> None:
        n = ThetaNeuron(theta=math.pi / 2.0, dt=1.0)
        assert n.step(0.0) == 1
        assert n.theta == -math.pi

    def test_negative_flow_exponential_overflow_is_rejected_without_mutation(self) -> None:
        n = ThetaNeuron(theta=0.25, dt=400.0)
        before = n.theta
        with pytest.raises(ValueError, match="exact-flow candidate"):
            n.step(-1.0)
        assert n.theta == before

    def test_negative_flow_singularity_wraps_and_reports_crossing(self) -> None:
        theta = 2.0
        y = math.tan(theta / 2.0)
        ratio = (y - 1.0) / (y + 1.0)
        n = ThetaNeuron(theta=theta, dt=-math.log(ratio) / 2.0)
        assert n.step(-1.0) == 1
        assert n.theta == -math.pi

    def test_deterministic(self) -> None:
        traces = []
        for _ in range(2):
            n = ThetaNeuron()
            trace = [(n.step(2.0), n.theta) for _ in range(200)]
            traces.append(trace)
        assert traces[0] == traces[1]
