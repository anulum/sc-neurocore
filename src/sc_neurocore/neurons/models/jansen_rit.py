# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class JansenRitUnit:
    """Jansen & Rit 1995 — neural mass model for EEG generation.

    6 ODEs: 3 populations (pyramidal, excitatory, inhibitory) x 2 states.
    """

    y0: float = 0.0
    y3: float = 0.0
    y1: float = 0.0
    y4: float = 0.0
    y2: float = 0.0
    y5: float = 0.0
    a_exc: float = 3.25
    b_exc: float = 22.0
    a_rate: float = 100.0
    b_rate: float = 50.0
    c: float = 135.0
    e0: float = 2.5
    v0: float = 6.0
    r: float = 0.56
    dt: float = 0.001

    def _sigmoid(self, x):
        return 2.0 * self.e0 / (1.0 + np.exp(self.r * (self.v0 - x)))

    def step(self, p_ext: float = 220.0) -> float:
        s1 = self._sigmoid(self.y1 - self.y2)
        s0 = self._sigmoid(self.c * 0.8 * self.y0)
        s2 = self._sigmoid(self.c * 0.25 * self.y0)
        dy0 = self.y3
        dy3 = self.a_exc * self.a_rate * s1 - 2.0 * self.a_rate * self.y3 - self.a_rate**2 * self.y0
        dy1 = self.y4
        dy4 = (
            self.a_exc * self.a_rate * (p_ext + self.c * 0.8 * s0)
            - 2.0 * self.a_rate * self.y4
            - self.a_rate**2 * self.y1
        )
        dy2 = self.y5
        dy5 = (
            self.b_exc * self.b_rate * self.c * 0.25 * s2
            - 2.0 * self.b_rate * self.y5
            - self.b_rate**2 * self.y2
        )
        self.y0 += dy0 * self.dt
        self.y3 += dy3 * self.dt
        self.y1 += dy1 * self.dt
        self.y4 += dy4 * self.dt
        self.y2 += dy2 * self.dt
        self.y5 += dy5 * self.dt
        return self.y1 - self.y2

    def reset(self):
        self.y0 = self.y1 = self.y2 = self.y3 = self.y4 = self.y5 = 0.0
