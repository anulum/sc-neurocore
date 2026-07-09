# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header


from dataclasses import dataclass
import numpy as np


@dataclass
class MyceliumLayer:
    """
    Fungal Computing Layer.
    Simulates a dynamic mycelial network that reinforces active paths.
    """

    n_nodes: int
    growth_rate: float = 0.1
    decay_rate: float = 0.05

    def __post_init__(self):
        # Conductance Matrix (Weights)
        self.conductance = np.random.uniform(0.1, 0.5, (self.n_nodes, self.n_nodes))
        # Remove self-loops
        np.fill_diagonal(self.conductance, 0)

    def step(self, inputs: np.ndarray) -> np.ndarray:
        """
        inputs: Activity at nodes.
        Returns: Propagated activity.
        """
        # Flux = Input * Conductance
        flux = np.dot(inputs, self.conductance)

        # Adaptation:
        # dG/dt = growth * Flux - decay * G
        # Edges with high flux grow thicker (more conductance).

        # We need "Flux through edge ij".
        # Flux_ij ~ Input_i * G_ij (simplified)
        # Or symmetric: (In_i + In_j) * G_ij

        # Matrix of inputs (broadcast)
        input_matrix = inputs[:, None] + inputs[None, :]  # (N, N) sum
        edge_flux = input_matrix * self.conductance

        delta_g = (self.growth_rate * edge_flux) - (self.decay_rate * self.conductance)
        self.conductance += delta_g
        self.conductance = np.clip(self.conductance, 0, 1.0)
        np.fill_diagonal(self.conductance, 0)

        return flux
