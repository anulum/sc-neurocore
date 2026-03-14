# SPDX-License-Identifier: AGPL-3.0-or-later
from typing import Any
import numpy as np
from dataclasses import dataclass


@dataclass
class ChaoticRNG:
    """
    Chaotic Random Number Generator using Logistic Map.
    x_{n+1} = r * x_n * (1 - x_n)

    Provides 'True' Randomness simulation (Deterministic Chaos)
    unlike linear PRNGs.
    """

    r: float = 4.0
    x: float = 0.5

    def __post_init__(self) -> None:
        # Burn-in to forget initial state
        for _ in range(100):
            self.x = self.r * self.x * (1.0 - self.x)

    def random(self, size: int) -> np.ndarray[Any, Any]:
        """
        Generate 'size' random floats [0, 1].
        """
        out = np.zeros(size)
        curr = self.x
        for i in range(size):
            curr = self.r * curr * (1.0 - curr)
            out[i] = curr

        self.x = curr
        return out

    def generate_bitstream(self, p: float, length: int) -> np.ndarray[Any, Any]:
        """
        Generate bitstream using chaotic source.
        """
        vals = self.random(length)
        return (vals < p).astype(np.uint8)
