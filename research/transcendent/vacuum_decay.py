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
class FalseVacuumField:
    """
    Simulates Scalar Field Vacuum Decay.
    Computing via Phase Transition Bubbles.
    """

    size: int
    false_vacuum_val: float = 0.0
    true_vacuum_val: float = 1.0
    barrier_height: float = 0.5

    def __post_init__(self):
        # Initialize in False Vacuum
        self.field = np.full((self.size, self.size), self.false_vacuum_val)

    def nucleate(self, x: int, y: int):
        """
        Injects a True Vacuum bubble (Logic Input 1).
        """
        if 0 <= x < self.size and 0 <= y < self.size:
            self.field[y, x] = self.true_vacuum_val

    def step(self):
        """
        Propagate bubbles (Phase Transition).
        True Vacuum expands into False Vacuum (Lower Energy).
        """
        # Simple Cellular Automaton rule for expansion
        # If neighbor is True, become True (Speed of Light expansion)

        # Create shifted views
        n = self.field
        top = np.roll(n, 1, axis=0)
        bottom = np.roll(n, -1, axis=0)
        left = np.roll(n, 1, axis=1)
        right = np.roll(n, -1, axis=1)

        # Max neighbor (if any neighbor is 1.0, become 1.0)
        # This simulates deterministic expansion
        max_neighbor = np.maximum.reduce([top, bottom, left, right])

        # Update: If neighbor is True Vacuum, flip self
        # (Simulating wall velocity = 1 grid unit per step)
        mask = max_neighbor > self.barrier_height
        self.field[mask] = self.true_vacuum_val

    def measure_energy(self) -> float:
        """
        Total energy released (Proportional to True Vacuum area).
        """
        return np.sum(self.field == self.true_vacuum_val)
