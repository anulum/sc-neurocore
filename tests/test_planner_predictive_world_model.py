# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPredictiveWorldModel from former test_planner.py

"""Focused suite: TestPredictiveWorldModel from former test_planner.py."""

from __future__ import annotations

from tests.planner_support import *  # noqa: F403


class TestPredictiveWorldModel:
    """Tests against the new sophisticated LGSSM-backed wrapper.

    Two prior tests (transition_matrix shape + row-normalisation)
    enforced a legacy `transition_matrix` design which was replaced
    by a proper Linear Gaussian SSM. They are removed because they
    were testing the wrong thing.

    Two more (`test_predict_next_state_bounded`) enforced the
    clip-to-[0,1] wrapper on the deterministic-linear implementation.
    Predictions can take any real value depending on the SSM dynamics,
    so the bounded-output assertion is also removed.
    """

    def test_construction_exposes_lgssm(self) -> None:
        """The wrapper holds a proper LinearGaussianSSM."""
        m = PredictiveWorldModel(state_dim=4, action_dim=2)
        assert m.model.state_dim == 4
        assert m.model.control_dim == 2
        assert m.model.obs_dim == 4

    def test_predict_next_state_shape(self) -> None:
        m = PredictiveWorldModel(state_dim=4, action_dim=2)
        s = np.array([0.5, 0.3, 0.8, 0.1])
        a = np.array([0.5, 0.5])
        ns = m.predict_next_state(s, a)
        assert ns.shape == (4,)

    def test_predict_next_state_obeys_ssm_dynamics(self) -> None:
        """Output must equal A·x + B·u (deterministic mean prediction)."""
        m = PredictiveWorldModel(state_dim=4, action_dim=2)
        s = np.array([0.5, 0.3, 0.8, 0.1])
        a = np.array([1.0, 1.0])
        ns = m.predict_next_state(s, a)
        expected = m.model.A @ s + m.model.B @ a
        np.testing.assert_allclose(ns, expected, atol=1e-12)

    def test_forecast_sequence(self) -> None:
        m = PredictiveWorldModel(state_dim=3, action_dim=1)
        s0 = np.array([0.5, 0.5, 0.5])
        actions = [np.array([0.5])] * 5
        traj = m.forecast(s0, actions)
        assert len(traj) == 5
        assert all(t.shape == (3,) for t in traj)
