# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class RulkovMapNeuron:
    """Rulkov 2001 — discrete map-based neuron (no ODE, O(1) per step).

    x[n+1] = f(x[n], y[n]) + I
    y[n+1] = y[n] - μ(x[n] + 1) + μσ
    Fast iteration, exhibits spiking and bursting.
    """

    x: float = -1.0
    y: float = -3.0
    alpha: float = 4.0
    sigma: float = -1.6
    mu: float = 0.001
    x_threshold: float = 0.0

    def step(self, current: float = 0.0) -> int:
        x_prev = self.x
        if self.x <= 0:
            x_new = self.alpha / (1.0 - self.x) + self.y + current
        elif self.x < self.alpha + self.y + current:
            x_new = self.alpha + self.y + current
        else:
            x_new = -1.0
        y_new = self.y - self.mu * (self.x + 1.0) + self.mu * self.sigma
        self.x = x_new
        self.y = y_new
        return 1 if (self.x >= self.x_threshold and x_prev < self.x_threshold) else 0

    def reset(self):
        self.x, self.y = -1.0, -3.0
