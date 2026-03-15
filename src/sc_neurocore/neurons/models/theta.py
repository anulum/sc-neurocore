# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class ThetaNeuron:
    """Theta neuron — canonical Type-I on the unit circle.

    dθ/dt = (1 - cos θ) + (1 + cos θ) · I
    Spike when θ crosses π.
    Ermentrout & Kopell 1986.
    """

    theta: float = 0.0
    dt: float = 0.01

    def step(self, current: float) -> int:
        theta_prev = self.theta
        dtheta = ((1.0 - np.cos(self.theta)) + (1.0 + np.cos(self.theta)) * current) * self.dt
        self.theta += dtheta
        # Wrap to [-π, π]
        self.theta = ((self.theta + np.pi) % (2 * np.pi)) - np.pi
        return 1 if (theta_prev < np.pi * 0.99 and self.theta >= np.pi * 0.99) else 0

    def reset(self):
        self.theta = 0.0
