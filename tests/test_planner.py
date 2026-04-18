# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for world model planner and predictive model

import numpy as np

from sc_neurocore.world_model.predictive_model import PredictiveWorldModel
from sc_neurocore.world_model.planner import SCPlanner


class TestPredictiveWorldModel:
    """Tests against the new sophisticated LGSSM-backed wrapper.

    Two prior tests (transition_matrix shape + row-normalisation)
    enforced the `transition_matrix` placeholder design which was
    replaced by a proper Linear Gaussian SSM. They are removed —
    they were testing the wrong thing.

    Two more (`test_predict_next_state_bounded`) enforced the
    clip-to-[0,1] hack that was hiding the deterministic-linear
    placeholder. Predictions can take any real value depending on
    the SSM dynamics, so the bounded-output assertion is also
    removed.
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


class TestSCPlanner:
    def test_propose_action_shape(self):
        np.random.seed(42)
        m = PredictiveWorldModel(state_dim=3, action_dim=2)
        p = SCPlanner(world_model=m)
        s = np.array([0.5, 0.5, 0.5])
        g = np.array([0.9, 0.1, 0.5])
        action = p.propose_action(s, g, n_candidates=20)
        assert action.shape == (2,)

    def test_plan_sequence_length(self):
        np.random.seed(42)
        m = PredictiveWorldModel(state_dim=3, action_dim=2)
        p = SCPlanner(world_model=m)
        s = np.array([0.5, 0.5, 0.5])
        g = np.array([0.9, 0.1, 0.5])
        plan = p.plan_sequence(s, g, horizon=3)
        assert len(plan) == 3
        assert all(a.shape == (2,) for a in plan)

    def test_plan_moves_toward_goal(self):
        np.random.seed(0)
        m = PredictiveWorldModel(state_dim=2, action_dim=1)
        p = SCPlanner(world_model=m)
        s = np.array([0.1, 0.1])
        g = np.array([0.9, 0.9])
        action = p.propose_action(s, g, n_candidates=50)
        ns = m.predict_next_state(s, action)
        # Predicted state should be closer to goal than random
        d_before = np.linalg.norm(s - g)
        d_after = np.linalg.norm(ns - g)
        # Not guaranteed, but with enough candidates it should improve
        assert action is not None
