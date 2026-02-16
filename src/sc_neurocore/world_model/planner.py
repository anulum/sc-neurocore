from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import List
from .predictive_model import PredictiveWorldModel


@dataclass
class SCPlanner:
    """
    A planner that uses a PredictiveWorldModel to select actions.
    """

    world_model: PredictiveWorldModel

    def propose_action(
        self, current_state: np.ndarray, goal_state: np.ndarray, n_candidates: int = 10
    ) -> np.ndarray:
        """
        Propose the best action among n_candidates based on predicted outcome.
        """
        best_action = None
        min_dist = float("inf")

        for _ in range(n_candidates):
            # Sample a random action
            candidate_action = np.random.uniform(0, 1, self.world_model.action_dim)

            # Predict next state
            predicted_state = self.world_model.predict_next_state(current_state, candidate_action)

            # Evaluate distance to goal
            dist = np.linalg.norm(predicted_state - goal_state)

            if dist < min_dist:
                min_dist = dist
                best_action = candidate_action

        return best_action

    def plan_sequence(
        self, current_state: np.ndarray, goal_state: np.ndarray, horizon: int = 5
    ) -> List[np.ndarray]:
        """
        Simple greedy planning for a sequence of actions.
        """
        plan = []
        curr_s = current_state
        for _ in range(horizon):
            action = self.propose_action(curr_s, goal_state)
            plan.append(action)
            curr_s = self.world_model.predict_next_state(curr_s, action)
        return plan
