# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class AdExNeuron:
    """Adaptive Exponential Integrate-and-Fire. Brette & Gerstner 2005.

    dv/dt = -(v - v_rest)/tau + delta_T * exp((v - v_rh)/delta_T) / tau - w/C + I/C
    dw/dt = (a * (v - v_rest) - w) / tau_w
    if v >= v_threshold: v = v_reset, w += b
    """

    v: float = -65.0
    w: float = 0.0
    v_rest: float = -65.0
    v_reset: float = -68.0
    v_threshold: float = -50.0
    v_rh: float = -55.0
    delta_t: float = 2.0
    tau: float = 20.0
    tau_w: float = 100.0
    a: float = 0.5
    b: float = 7.0
    c_m: float = 200.0
    dt: float = 0.1

    def step(self, current: float) -> int:
        exp_term = self.delta_t * np.exp(np.clip((self.v - self.v_rh) / self.delta_t, -20.0, 20.0))
        dv = (-(self.v - self.v_rest) + exp_term - self.w + current) / self.tau * self.dt
        dw = (self.a * (self.v - self.v_rest) - self.w) / self.tau_w * self.dt

        self.v += dv
        self.w += dw

        if self.v >= self.v_threshold:
            self.v = self.v_reset
            self.w += self.b
            return 1
        return 0

    def reset(self):
        self.v = self.v_rest
        self.w = 0.0
