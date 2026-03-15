# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FitzHughNagumoNeuron:
    """FitzHugh-Nagumo 1961 — 2D qualitative spike model.

    dv/dt = v - v³/3 - w + I
    dw/dt = ε(v + a - bw)
    """

    v: float = -1.0
    w: float = -0.5
    a: float = 0.7
    b: float = 0.8
    epsilon: float = 0.08
    dt: float = 0.1
    v_threshold: float = 1.0

    def step(self, current: float) -> int:
        v_prev = self.v
        dv = (self.v - self.v**3 / 3.0 - self.w + current) * self.dt
        dw = self.epsilon * (self.v + self.a - self.b * self.w) * self.dt
        self.v += dv
        self.w += dw
        return 1 if (self.v >= self.v_threshold and v_prev < self.v_threshold) else 0

    def reset(self):
        self.v = -1.0
        self.w = -0.5
