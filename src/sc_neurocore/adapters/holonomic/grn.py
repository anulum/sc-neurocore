# SPDX-License-Identifier: AGPL-3.0-or-later
from typing import Any
from dataclasses import dataclass
import numpy as np


@dataclass
class GeneticRegulatoryLayer:
    """
    Bio-Hybrid Layer.
    Neural Activity -> Gene Expression (Protein) -> Neural Param Modulation.
    """

    n_neurons: int
    production_rate: float = 0.01
    decay_rate: float = 0.005

    def __post_init__(self):  # type: ignore
        self.protein_levels = np.zeros(self.n_neurons)

    def step(self, spikes: np.ndarray[Any, Any]):  # type: ignore
        """
        Update protein levels based on spike activity.
        """
        # dP/dt = alpha * spikes - beta * P
        delta = (self.production_rate * spikes) - (self.decay_rate * self.protein_levels)
        self.protein_levels += delta
        self.protein_levels = np.clip(self.protein_levels, 0, 10.0)

    def get_threshold_modulators(self) -> np.ndarray[Any, Any]:
        """
        Protein acts as inhibitor: Higher protein -> Higher threshold.
        """
        return self.protein_levels
