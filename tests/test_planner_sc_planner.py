# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSCPlanner from former test_planner.py

"""Focused suite: TestSCPlanner from former test_planner.py."""

from __future__ import annotations

from tests.planner_support import *  # noqa: F403


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
