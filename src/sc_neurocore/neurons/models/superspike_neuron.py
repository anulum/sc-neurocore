# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Zenke & Ganguli 2018 — LIF with SuperSpike surrogate

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SuperSpikeNeuron:
    """Zenke & Ganguli 2018 — LIF with SuperSpike surrogate gradient.

    Uses Van Rossum filtered eligibility traces and a smooth surrogate
    gradient sigma'(V) = 1/(beta * |V - V_th| + 1)^2.

    Reference: Zenke, F. & Ganguli, S. (2018). Neural Comput. 30:1514–1541.
    """

    v: float = 0.0
    trace: float = 0.0  # Van Rossum eligibility trace
    tau_m: float = 10.0
    tau_e: float = 10.0  # eligibility filter time constant
    v_threshold: float = 1.0
    v_reset: float = 0.0
    beta_sg: float = 10.0  # surrogate gradient sharpness
    dt: float = 1.0
    alpha_m: float = field(init=False)
    alpha_e: float = field(init=False)

    def __post_init__(self) -> None:
        self.alpha_m = np.exp(-self.dt / self.tau_m)
        self.alpha_e = np.exp(-self.dt / self.tau_e)

    def surrogate_grad(self) -> float:
        return 1.0 / (self.beta_sg * abs(self.v - self.v_threshold) + 1.0) ** 2

    def step(self, current: float) -> int:
        self.v = self.alpha_m * self.v + current
        sg = self.surrogate_grad()
        self.trace = self.alpha_e * self.trace + sg
        if self.v >= self.v_threshold:
            self.v = self.v_reset
            return 1
        return 0

    def reset(self) -> None:
        self.v, self.trace = 0.0, 0.0
