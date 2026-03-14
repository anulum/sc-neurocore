# SPDX-License-Identifier: AGPL-3.0-or-later
from dataclasses import dataclass
import numpy as np
from ..layers.sc_learning_layer import SCLearningLayer


@dataclass
class EWC_SCLayer(SCLearningLayer):
    """
    Lifelong Learning Layer using Elastic Weight Consolidation (Approx).
    """

    ewc_lambda: float = 10.0  # Strength of constraint

    def __post_init__(self) -> None:
        super().__post_init__()
        self.fisher_info = np.zeros((self.n_neurons, self.n_inputs))
        self.star_weights = np.zeros((self.n_neurons, self.n_inputs))

    def consolidate_task(self) -> None:
        """
        Call after finishing a task.
        Calculate Fisher Info (Importance) and freeze 'star' weights.
        """
        # In SC, Fisher Info approx ~ Activity * Plasticity
        # Weights that changed a lot or are high are often important.
        # Simplified: Importance = Current Weight Magnitude (Hebbian)

        current_w = self.get_weights()
        self.star_weights = current_w.copy()
        # Assume all non-zero weights are somewhat important
        self.fisher_info = current_w.copy()

    def apply_ewc_penalty(self) -> None:
        """
        This would be called during the learning loop.
        Instead of gradient descent, we modify STDP probability.
        """
        # If we try to move W away from W_star, probability decreases.
        pass  # Concept implementation logic resides in the custom synapse step usually.
        # For this demo, we assume the 'consolidate' action sets the state.
