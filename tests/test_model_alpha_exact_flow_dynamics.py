# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestExactFlowDynamics from former test_model_alpha.py

"""Focused suite: TestExactFlowDynamics from former test_model_alpha.py."""

from __future__ import annotations

from tests.model_alpha_support import *  # noqa: F403

class TestExactFlowDynamics:
    """Each step is the exact constant-input flow, never an Euler step."""

    def test_filter_matches_exact_alpha_cascade(self) -> None:
        n = AlphaNeuron(a_exc=0.25, i_exc=0.1, v_threshold=100.0, dt=0.5)
        steady = 5.0 * 2.0
        decay = math.exp(-0.5 / 5.0)
        expected_a = steady + (0.25 - steady) * decay
        expected_i = steady + decay * ((0.1 - steady) + (0.25 - steady) * 0.5 / 5.0)
        assert n.step(2.0) == 0
        assert n.a_exc == pytest.approx(expected_a, rel=0.0, abs=1e-14)
        assert n.i_exc == pytest.approx(expected_i, rel=0.0, abs=1e-14)

    def test_step_is_not_forward_euler(self) -> None:
        n = AlphaNeuron(v=0.5, v_threshold=100.0, dt=0.5)
        n.step(0.0)
        euler_v = 0.5 + (-(0.5 - 0.0)) / 20.0 * 0.5
        assert abs(n.v - euler_v) > 1.0e-4

    def test_equal_time_constant_limit_is_analytic(self) -> None:
        n = AlphaNeuron(i_exc=0.3, a_exc=0.2, tau_v=20.0, tau_exc=20.0, v_threshold=100.0, dt=0.5)
        rate = 1.0 / 20.0
        decay = math.exp(-0.5 / 20.0)
        contribution = rate * decay * (0.3 * 0.5 + 0.2 * 0.5 * 0.5 / (2.0 * 20.0))
        expected_v = n.v * decay + contribution
        assert n.step(0.0) == 0
        assert n.v == pytest.approx(expected_v, rel=0.0, abs=1e-14)

    def test_large_timestep_remains_bounded(self) -> None:
        n = AlphaNeuron(v=0.5, tau_v=0.04, tau_exc=0.04, tau_inh=0.04, dt=1.0)
        assert n.step(0.0) == 0
        assert n.v == pytest.approx(n.v_rest, rel=0.0, abs=1e-8)

    def test_steady_state_is_a_fixed_point(self) -> None:
        n = AlphaNeuron(v_threshold=100.0)
        n.v = n.v_rest + 5.0 * 2.0 - 10.0 * 1.0
        n.a_exc = 5.0 * 2.0
        n.i_exc = 5.0 * 2.0
        n.a_inh = 10.0 * 1.0
        n.i_inh = 10.0 * 1.0
        assert n.step(2.0, 1.0) == 0
        assert n.v == pytest.approx(0.0, rel=0.0, abs=1e-13)
