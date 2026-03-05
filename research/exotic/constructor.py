# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np
from dataclasses import dataclass

@dataclass
class ConstructorCell:
    """
    Simulates a Universal Constructor (Von Neumann).
    A cell capable of replicating itself and evolving structure.
    """
    id: int
    blueprint: np.ndarray # Instructions

    def replicate(self) -> 'ConstructorCell':
        """
        Creates a copy of itself based on the blueprint.
        """
        # Simulation of self-replication
        return ConstructorCell(id=self.id + 1, blueprint=self.blueprint.copy())

    def mutate_blueprint(self, rate: float = 0.05):
        """
        Evolves instructions.
        """
        mask = np.random.random(self.blueprint.shape) < rate
        self.blueprint[mask] = 1 - self.blueprint[mask]
