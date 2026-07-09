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
class CatomLattice:
    """
    Programmable Matter Simulation (Claytronics).
    Catoms rearrange to form optimal topology.
    """

    size: int

    def __post_init__(self):
        # 1D Lattice for simplicity. [0, 1, 2, 3...]
        self.catom_ids = np.arange(self.size)
        # Random initial data/load on catoms
        self.load = np.random.random(self.size)

    def reconfigure(self):
        """
        Catoms swap positions to group high-load units together (Heat dissipation logic).
        """
        # Bubble sort style swap based on 'load'
        for i in range(self.size - 1):
            if self.load[i] < self.load[i + 1]:
                # Swap physically
                self.load[i], self.load[i + 1] = self.load[i + 1], self.load[i]
                self.catom_ids[i], self.catom_ids[i + 1] = self.catom_ids[i + 1], self.catom_ids[i]

    def get_topology(self) -> np.ndarray:
        return self.catom_ids
