# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Bio-Hybrid Layer

from dataclasses import dataclass
from typing import Any

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

    def __post_init__(self) -> None:
        self.protein_levels = np.zeros(self.n_neurons)

    def step(self, spikes: np.ndarray[Any, Any]) -> None:
        """
        Update protein levels based on spike activity.
        """
        # dP/dt = alpha * spikes - beta * P
        delta = (self.production_rate * spikes) - (self.decay_rate * self.protein_levels)
        self.protein_levels += delta
        self.protein_levels = np.clip(self.protein_levels, 0, 10.0)  # type: ignore[assignment]

    def get_threshold_modulators(self) -> np.ndarray[Any, Any]:
        """
        Protein acts as inhibitor: Higher protein -> Higher threshold.
        """
        return self.protein_levels
