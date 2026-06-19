# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Hasani et al. 2021 — liquid time-constant neuron

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LiquidTimeConstantNeuron:
    """Hasani et al. 2021 — liquid time-constant neuron.

    dx/dt = -(1/tau(x,I)) * x + (1/tau(x,I)) * f(x,I)
    where tau depends on input, making the neuron's time constant
    adaptive and input-driven.

    Reference: Hasani, R. et al. (2021). Proc. AAAI Conf. Artif. Intell. 35(9):7657–7666.
    """

    x: float = 0.0
    tau_base: float = 10.0
    w_tau: float = -0.5  # input→tau coupling
    w_x: float = 0.8
    w_in: float = 1.0
    bias: float = 0.0
    v_threshold: float = 1.0
    dt: float = 1.0

    def step(self, current: float) -> int:
        # Hasani 2021 Eq. 2: input-dependent time constant
        tau = self.tau_base * (1.0 / (1.0 + np.exp(-(self.w_tau * current + self.bias))))
        tau = max(tau, 0.1)
        f_target = np.tanh(self.w_x * self.x + self.w_in * current)
        self.x += self.dt / tau * (-self.x + f_target)
        if self.x >= self.v_threshold:
            self.x = 0.0
            return 1
        return 0

    def reset(self) -> None:
        self.x = 0.0
