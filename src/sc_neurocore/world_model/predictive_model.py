
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class PredictiveWorldModel:
    """
    A stochastic predictive world model.
    Predicts state_next = f(state_curr, action).
    """
    state_dim: int
    action_dim: int
    
    def __post_init__(self):
        # Internal transition weights (simplified)
        self.transition_matrix = np.random.uniform(0, 1, (self.state_dim, self.state_dim + self.action_dim))
        # Normalize rows to represent probabilities
        row_sums = self.transition_matrix.sum(axis=1)
        self.transition_matrix /= row_sums[:, np.newaxis]

    def predict_next_state(self, current_state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """
        Predicts the next state given current state and action.
        Inputs:
            current_state: (state_dim,) array of probabilities.
            action: (action_dim,) array of probabilities.
        Returns:
            next_state: (state_dim,) predicted probabilities.
        """
        # Concatenate state and action
        combined_input = np.concatenate([current_state, action])
        
        # Linear transition in probability domain
        next_state = np.dot(self.transition_matrix, combined_input)
        
        # Clip to ensure valid probabilities
        return np.clip(next_state, 0, 1)

    def forecast(self, initial_state: np.ndarray, actions: list[np.ndarray]) -> list[np.ndarray]:
        """
        Forecast multiple steps ahead given a sequence of actions.
        """
        trajectory = []
        curr_state = initial_state
        for act in actions:
            curr_state = self.predict_next_state(curr_state, act)
            trajectory.append(curr_state)
        return trajectory
