# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Hasani et al. 2022 — closed-form continuous-depth neuron

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class ClosedFormContinuousNeuron:
    """Hasani et al. 2022 — closed-form continuous-depth neuron (CfC).

    Analytical solution of the LTC ODE between timesteps:
    x(t+dt) = x(t)*exp(-dt/tau_eff) + f_target*(1 - exp(-dt/tau_eff))
    where tau_eff and f_target depend on input.

    Reference: Canavier, C.C. et al. (1993). Biophys. J. 65:2373–2382.
    """

    x: float = 0.0
    w_tau: float = -0.5
    w_x: float = 0.8
    w_in: float = 1.0
    tau_base: float = 10.0
    bias: float = 0.0
    v_threshold: float = 1.0
    dt: float = 1.0

    def __post_init__(self) -> None:
        for name in ("x", "w_tau", "w_x", "w_in", "bias"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        for name in ("tau_base", "v_threshold", "dt"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")

    @staticmethod
    def _sigmoid(value: float) -> float:
        if value >= 0.0:
            z = math.exp(-value)
            return 1.0 / (1.0 + z)
        z = math.exp(value)
        return z / (1.0 + z)

    def step(self, current: float) -> int:
        if not math.isfinite(current):
            raise ValueError("current must be finite")

        sigma_tau = self._sigmoid(self.w_tau * current + self.bias)
        tau_eff = max(self.tau_base * sigma_tau, 0.1)
        f_target = math.tanh(self.w_x * self.x + self.w_in * current)
        decay = math.exp(-self.dt / tau_eff)
        self.x = self.x * decay + f_target * (1.0 - decay)
        if self.x >= self.v_threshold:
            self.x = 0.0
            return 1
        return 0

    def reset(self) -> None:
        self.x = 0.0
