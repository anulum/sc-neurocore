# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for world-model planner contracts

"""Contracts for PredictiveWorldModel and SCPlanner interactions."""

from __future__ import annotations

import numpy as np


def test_predictive_world_model_forecasts_action_sequence() -> None:
    from sc_neurocore.world_model.predictive_model import PredictiveWorldModel

    model = PredictiveWorldModel(state_dim=4, action_dim=2)

    prediction = model.predict_next_state(current_state=np.zeros(4), action=np.zeros(2))
    trajectory = model.forecast(initial_state=np.zeros(4), actions=[np.zeros(2), np.ones(2)])

    assert prediction.shape == (4,)
    assert len(trajectory) == 2


def test_planner_returns_action_and_sequence_for_goal() -> None:
    from sc_neurocore.world_model.planner import SCPlanner
    from sc_neurocore.world_model.predictive_model import PredictiveWorldModel

    planner = SCPlanner(PredictiveWorldModel(state_dim=4, action_dim=2))

    action = planner.propose_action(
        current_state=np.zeros(4),
        goal_state=np.ones(4),
        n_candidates=5,
    )
    plan = planner.plan_sequence(
        current_state=np.zeros(4),
        goal_state=np.ones(4),
        horizon=3,
    )

    assert isinstance(action, np.ndarray)
    assert len(plan) > 0
