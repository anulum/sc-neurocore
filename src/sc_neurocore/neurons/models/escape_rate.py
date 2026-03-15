# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class EscapeRateNeuron:
    """Gerstner 2000 — stochastic threshold (escape noise model)."""

    v: float = -70.0
    v_rest: float = -70.0
    v_reset: float = -70.0
    v_threshold: float = -50.0
    tau_m: float = 10.0
    rho_0: float = 0.001
    delta_u: float = 3.0
    resistance: float = 1.0
    dt: float = 1.0

    def step(self, current: float) -> int:
        self.v += (-(self.v - self.v_rest) + self.resistance * current) / self.tau_m * self.dt
        rate = self.rho_0 * np.exp((self.v - self.v_threshold) / self.delta_u)
        p_spike = rate * self.dt
        if np.random.random() < p_spike:
            self.v = self.v_reset
            return 1
        return 0

    def reset(self):
        self.v = self.v_rest
