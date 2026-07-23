# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestThetaValidation from former test_model_theta.py

"""Focused suite: TestThetaValidation from former test_model_theta.py."""

from __future__ import annotations

from tests.model_theta_support import *  # noqa: F403

class TestThetaValidation:
    @pytest.mark.parametrize("theta", [np.nan, np.inf, -np.inf])
    def test_rejects_non_finite_initial_phase(self, theta: float) -> None:
        with pytest.raises(ValueError, match="theta"):
            ThetaNeuron(theta=theta)

    @pytest.mark.parametrize("dt", [0.0, -1.0, np.nan, np.inf])
    def test_rejects_non_positive_or_non_finite_dt(self, dt: float) -> None:
        with pytest.raises(ValueError, match="dt"):
            ThetaNeuron(dt=dt)

    @pytest.mark.parametrize("current", [np.nan, np.inf, -np.inf])
    def test_rejects_non_finite_current_before_phase_mutation(self, current: float) -> None:
        n = ThetaNeuron(theta=0.25)
        before = n.theta
        with pytest.raises(ValueError, match="current"):
            n.step(current)
        assert n.theta == before

    def test_initial_phase_is_wrapped_to_compact_circle(self) -> None:
        n = ThetaNeuron(theta=4.0 * np.pi + 0.5)
        assert -np.pi <= n.theta <= np.pi
        assert abs(n.theta - 0.5) < 1e-12

    def test_rejects_non_finite_exact_candidate_before_state_mutation(self) -> None:
        n = ThetaNeuron(theta=0.25, dt=1.0e308)
        before = n.theta
        with pytest.raises(ValueError, match="exact-flow candidate"):
            n.step(-1.0e308)
        assert n.theta == before

    @pytest.mark.parametrize("field", ["theta", "dt"])
    def test_rejects_corrupted_runtime_state_before_phase_mutation(self, field: str) -> None:
        n = ThetaNeuron(theta=0.25)
        before = n.theta
        setattr(n, field, np.nan)
        with pytest.raises(ValueError, match="runtime"):
            n.step(1.0)
        if field != "theta":
            assert n.theta == before

    def test_rejects_runtime_dt_that_is_no_longer_positive(self) -> None:
        n = ThetaNeuron(theta=0.25)
        before = n.theta
        n.dt = 0.0
        with pytest.raises(ValueError, match="runtime"):
            n.step(1.0)
        assert n.theta == before
