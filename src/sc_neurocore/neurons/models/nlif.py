# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class NonlinearLIFNeuron:
    """Nonlinear LIF with cubic term. Touboul & Brette 2008.

    C dV/dt = a*(V - V_rest)*(V - V_crit) - w + I
    dw/dt = (b*(V - V_rest) - w) / tau_w
    """

    v: float = -65.0
    w: float = 0.0
    v_rest: float = -65.0
    v_crit: float = -40.0
    v_threshold: float = -20.0
    v_reset: float = -65.0
    a: float = 0.04
    b: float = 0.5
    tau_w: float = 100.0
    c_m: float = 1.0
    dt: float = 0.1

    def step(self, current: float) -> int:
        cubic = self.a * (self.v - self.v_rest) * (self.v - self.v_crit)
        dv = (cubic - self.w + current) / self.c_m * self.dt
        dw = (self.b * (self.v - self.v_rest) - self.w) / self.tau_w * self.dt
        self.v += dv
        self.w += dw
        if self.v >= self.v_threshold:
            self.v = self.v_reset
            return 1
        return 0

    def reset(self):
        self.v = self.v_rest
        self.w = 0.0
