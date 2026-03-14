# SPDX-License-Identifier: AGPL-3.0-or-later
from typing import Any
import numpy as np


class StochasticHeatSolver:
    """
    Solves 1D Heat Equation using Stochastic Random Walks (Feynman-Kac).
    """

    def __init__(self, length: int, num_walkers: int, alpha: float):
        self.length = length
        self.walkers = np.random.randint(0, length, num_walkers)
        self.alpha = alpha

    def step(self) -> None:
        """
        Move walkers.
        """
        # Random step -1, 0, 1
        steps = np.random.choice([-1, 0, 1], size=len(self.walkers), p=[0.25, 0.5, 0.25])
        self.walkers += steps

        # Boundary conditions (Reflective)
        self.walkers = np.clip(self.walkers, 0, self.length - 1)

    def get_temperature_profile(self) -> np.ndarray[Any, Any]:
        """
        Convert walker density to temperature.
        """
        density, _ = np.histogram(self.walkers, bins=self.length, range=(0, self.length))
        return density / len(self.walkers)
