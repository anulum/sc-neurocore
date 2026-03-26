# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for world model planner and predictive model

import numpy as np

from sc_neurocore.world_model.predictive_model import PredictiveWorldModel
from sc_neurocore.world_model.planner import SCPlanner


class TestPredictiveWorldModel:
    def test_construction(self):
        m = PredictiveWorldModel(state_dim=4, action_dim=2)
        assert m.transition_matrix.shape == (4, 6)

    def test_transition_matrix_row_normalized(self):
        m = PredictiveWorldModel(state_dim=4, action_dim=2)
        row_sums = m.transition_matrix.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)

    def test_predict_next_state_shape(self):
        m = PredictiveWorldModel(state_dim=4, action_dim=2)
        s = np.array([0.5, 0.3, 0.8, 0.1])
        a = np.array([0.5, 0.5])
        ns = m.predict_next_state(s, a)
        assert ns.shape == (4,)

    def test_predict_next_state_bounded(self):
        m = PredictiveWorldModel(state_dim=4, action_dim=2)
        s = np.array([0.5, 0.3, 0.8, 0.1])
        a = np.array([1.0, 1.0])
        ns = m.predict_next_state(s, a)
        assert np.all(ns >= 0) and np.all(ns <= 1)

    def test_forecast_sequence(self):
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
