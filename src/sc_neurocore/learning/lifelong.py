# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Lifelong Learning Layer using Elastic Weight

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

    def apply_ewc_penalty(self, step_size: float = 0.01) -> float:
        """Push weights back toward consolidated values, weighted by Fisher info.

        Kirkpatrick et al. 2017, adapted to SC/STDP setting.
        Penalty gradient per synapse: F_i * (w_i - w_star_i).

        Parameters
        ----------
        step_size : float
            Fraction of penalty gradient to apply per call.

        Returns
        -------
        float
            Total penalty magnitude (for logging).
        """
        current_w = self.get_weights()
        delta = current_w - self.star_weights
        penalty_grad = self.fisher_info * delta
        correction = self.ewc_lambda * step_size * penalty_grad
        new_w = np.clip(current_w - correction, self.w_min, self.w_max)

        for i in range(self.n_neurons):
            for j in range(self.n_inputs):
                self.synapses[i][j].w = float(new_w[i, j])

        return float(np.sum(np.abs(penalty_grad)))
