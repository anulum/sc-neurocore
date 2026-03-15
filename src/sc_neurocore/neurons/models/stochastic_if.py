# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class StochasticIFNeuron:
    """Brunel & Hakim 1999 — Ornstein-Uhlenbeck driven IF."""

    v: float = -70.0
    v_rest: float = -70.0
    v_reset: float = -70.0
    v_threshold: float = -50.0
    tau_m: float = 20.0
    mu: float = 0.0
    sigma: float = 3.0
    dt: float = 1.0

    def step(self, current: float) -> int:
        noise = self.sigma * np.sqrt(self.dt / self.tau_m) * np.random.randn()
        self.v += (-(self.v - self.v_rest) + self.mu + current) / self.tau_m * self.dt + noise
        if self.v >= self.v_threshold:
            self.v = self.v_reset
            return 1
        return 0

    def reset(self):
        self.v = self.v_rest
