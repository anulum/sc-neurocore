# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class FitzHughRinzelNeuron:
    """FitzHugh 1976 / Rinzel 1987 — FHN + slow variable for bursting."""

    v: float = -1.0
    w: float = -0.5
    y: float = 0.0
    a: float = 0.7
    b: float = 0.8
    c: float = -0.775
    d: float = 1.0
    delta: float = 0.08
    mu: float = 0.0001
    dt: float = 0.1
    v_threshold: float = 1.0

    def step(self, current: float) -> int:
        v_prev = self.v
        dv = (self.v - self.v**3 / 3.0 - self.w + self.y + current) * self.dt
        dw = self.delta * (self.a + self.v - self.b * self.w) * self.dt
        dy = self.mu * (self.c - self.v - self.d * self.y) * self.dt
        self.v += dv
        self.w += dw
        self.y += dy
        return 1 if (self.v >= self.v_threshold and v_prev < self.v_threshold) else 0

    def reset(self):
        self.v, self.w, self.y = -1.0, -0.5, 0.0
