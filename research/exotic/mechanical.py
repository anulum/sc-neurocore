# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header


import numpy as np
from dataclasses import dataclass


@dataclass
class MechanicalLatticeLayer:
    """
    Mechanical Neural Network.
    Computing with Stiffness (k) and Displacement (x).
    F = K x
    """

    n_nodes: int
    learning_rate: float = 0.01

    def __post_init__(self):
        # Stiffness matrix (Symmetric)
        self.K = np.random.uniform(0.5, 1.5, (self.n_nodes, self.n_nodes))
        np.fill_diagonal(self.K, 0)  # No self-springs

        # Displacements
        self.x = np.zeros(self.n_nodes)

    def relax(self, inputs: np.ndarray, clamped_nodes: list):
        """
        Solve equilibrium: Sum(Forces) = 0.
        F_i = Sum_j (K_ij * (x_j - x_i)) + Input_Force_i = 0
        """
        # Iterative relaxation (Gradient Descent on Energy)
        # Energy = 0.5 * Sum K_ij (x_i - x_j)^2 - Sum F_i x_i

        # Simple iterative update:
        # x_i_new = (Sum K_ij x_j + F_i) / Sum K_ij

        forces = inputs.copy()

        for _ in range(50):  # Relaxation steps
            for i in range(self.n_nodes):
                if i in clamped_nodes:
                    continue

                k_sum = np.sum(self.K[i])
                if k_sum == 0:
                    continue

                # Weighted avg of neighbors
                neighbor_force = np.dot(self.K[i], self.x)
                self.x[i] = (neighbor_force + forces[i]) / k_sum

    def train(self):
        """
        Adjust stiffness to minimize stress?
        Or 'Learning': Active springs get stiffer.
        """
        # Calculate strain (x_i - x_j)^2
        # Use simple rule: K += alpha * (x_i - x_j)^2
        # (This is like Hebbian: correlated movement reinforces connection)

        x_diff = self.x[:, None] - self.x[None, :]
        strain = x_diff**2

        self.K += self.learning_rate * strain
        self.K = np.clip(self.K, 0.1, 2.0)
        np.fill_diagonal(self.K, 0)
