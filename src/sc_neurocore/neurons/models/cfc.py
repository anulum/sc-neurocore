# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class ClosedFormContinuousNeuron:
    """Hasani et al. 2022 — closed-form continuous-depth neuron (CfC).

    Analytical solution of the LTC ODE between timesteps:
    x(t+dt) = x(t)*exp(-dt/tau_eff) + f_target*(1 - exp(-dt/tau_eff))
    where tau_eff and f_target depend on input.
    """

    x: float = 0.0
    w_tau: float = -0.5
    w_x: float = 0.8
    w_in: float = 1.0
    tau_base: float = 10.0
    bias: float = 0.0
    v_threshold: float = 1.0
    dt: float = 1.0

    def step(self, current: float) -> int:
        sigma_tau = 1.0 / (1.0 + np.exp(-(self.w_tau * current + self.bias)))
        tau_eff = max(self.tau_base * sigma_tau, 0.1)
        f_target = np.tanh(self.w_x * self.x + self.w_in * current)
        decay = np.exp(-self.dt / tau_eff)
        self.x = self.x * decay + f_target * (1.0 - decay)
        if self.x >= self.v_threshold:
            self.x = 0.0
            return 1
        return 0

    def reset(self):
        self.x = 0.0
