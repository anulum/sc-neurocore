# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class IntegerQIFNeuron:
    """Lo et al. 2021 — fixed-point quadratic integrate-and-fire.

    V[t+1] = V[t] + (V[t]^2 >> k) + I, all integer arithmetic.
    """

    v: int = 0
    k: int = 6  # right-shift for V^2
    v_threshold: int = 1024
    v_reset: int = -1024
    v_min: int = -2048

    def step(self, current: int) -> int:
        self.v = max(self.v_min, self.v + (self.v * self.v >> self.k) + current)
        if self.v >= self.v_threshold:
            self.v = self.v_reset
            return 1
        return 0

    def reset(self):
        self.v = 0
