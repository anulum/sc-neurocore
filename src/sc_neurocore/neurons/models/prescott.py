# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Prescott 2008 — Type I/II/III excitability via M-current

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class PrescottNeuron:
    """Prescott 2008 — Type I/II/III excitability via M-current tuning."""

    v: float = -65.0
    w: float = 0.0
    g_fast: float = 20.0
    g_slow: float = 20.0
    g_l: float = 2.0
    e_fast: float = 50.0
    e_slow: float = -100.0
    e_l: float = -70.0
    beta_w: float = -21.0
    gamma_w: float = 15.0
    tau_w: float = 100.0
    phi: float = 0.15
    dt: float = 0.1
    v_threshold: float = -20.0

    def step(self, current: float) -> int:
        v_prev = self.v
        m_inf = 1.0 / (1.0 + np.exp(-(self.v + 20.0) / 15.0))
        w_inf = 1.0 / (1.0 + np.exp(-(self.v - self.beta_w) / self.gamma_w))
        i_fast = self.g_fast * m_inf * (self.v - self.e_fast)
        i_slow = self.g_slow * self.w * (self.v - self.e_slow)
        i_l = self.g_l * (self.v - self.e_l)
        self.v += (-i_fast - i_slow - i_l + current) * self.dt
        self.w += self.phi * (w_inf - self.w) / self.tau_w * self.dt
        return 1 if (self.v >= self.v_threshold and v_prev < self.v_threshold) else 0

    def reset(self) -> None:
        self.v = -65.0
        self.w = 0.0
