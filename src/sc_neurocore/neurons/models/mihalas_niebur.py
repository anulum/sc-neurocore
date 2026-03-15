# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class MihalasNieburNeuron:
    """Mihalas-Niebur Generalized IF — captures 20 spike patterns.

    Mihalas & Niebur 2009. Multiple internal thresholds and
    adaptation currents enable tonic/phasic/burst/accommodation patterns.
    """

    v: float = 0.0
    theta: float = 1.0
    i1: float = 0.0
    i2: float = 0.0
    v_rest: float = 0.0
    v_reset: float = 0.0
    theta_reset: float = 1.0
    theta_inf: float = 1.0
    tau_v: float = 10.0
    tau_theta: float = 100.0
    tau_1: float = 10.0
    tau_2: float = 200.0
    a: float = 0.0
    b: float = 0.0
    r1: float = 0.0
    r2: float = 0.0
    dt: float = 1.0

    def step(self, current: float) -> int:
        dv = (-(self.v - self.v_rest) + self.i1 + self.i2 + current) / self.tau_v * self.dt
        dtheta = (
            (self.theta_inf - self.theta + self.a * (self.v - self.v_rest))
            / self.tau_theta
            * self.dt
        )
        di1 = -self.i1 / self.tau_1 * self.dt
        di2 = -self.i2 / self.tau_2 * self.dt
        self.v += dv
        self.theta += dtheta
        self.i1 += di1
        self.i2 += di2

        if self.v >= self.theta:
            self.v = self.v_reset
            self.theta = max(self.theta, self.theta_reset)
            self.i1 += self.r1
            self.i2 += self.r2
            return 1
        return 0

    def reset(self):
        self.v = self.v_rest
        self.theta = self.theta_reset
        self.i1 = 0.0
        self.i2 = 0.0
