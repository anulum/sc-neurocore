# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class WongWangUnit:
    """Wong & Wang 2006 — reduced decision-making attractor model."""

    s1: float = 0.1
    s2: float = 0.1
    tau_s: float = 0.1
    gamma: float = 0.641
    j_n: float = 0.2609
    j_cross: float = 0.0497
    i_0: float = 0.3255
    sigma: float = 0.02
    dt: float = 0.001

    def _phi(self, i_syn):
        a, b, d = 270.0, 108.0, 0.154
        x = a * i_syn - b
        if abs(x) < 1e-6:
            return 1.0 / d
        return x / (1.0 - np.exp(-d * x))

    def step(self, stim1: float = 0.0, stim2: float = 0.0) -> tuple:
        i1 = (
            self.j_n * self.s1
            - self.j_cross * self.s2
            + self.i_0
            + stim1
            + self.sigma * np.random.randn()
        )
        i2 = (
            self.j_n * self.s2
            - self.j_cross * self.s1
            + self.i_0
            + stim2
            + self.sigma * np.random.randn()
        )
        r1, r2 = self._phi(i1), self._phi(i2)
        self.s1 += (-self.s1 / self.tau_s + (1.0 - self.s1) * self.gamma * r1) * self.dt
        self.s2 += (-self.s2 / self.tau_s + (1.0 - self.s2) * self.gamma * r2) * self.dt
        self.s1 = np.clip(self.s1, 0.0, 1.0)
        self.s2 = np.clip(self.s2, 0.0, 1.0)
        return (r1, r2)

    def reset(self):
        self.s1, self.s2 = 0.1, 0.1
