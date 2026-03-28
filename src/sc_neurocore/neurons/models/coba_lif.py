# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Conductance-based LIF — Destexhe et al. 2001

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class COBALIFNeuron:
    """Conductance-based LIF — Destexhe et al. 2001.

    C dV/dt = -g_L(V - E_L) - g_e(V - E_e) - g_i(V - E_i) + I
    dg_e/dt = -g_e / tau_e
    dg_i/dt = -g_i / tau_i
    """

    v: float = -65.0
    g_e: float = 0.0
    g_i: float = 0.0
    c_m: float = 200.0
    g_l: float = 10.0
    e_l: float = -65.0
    e_e: float = 0.0
    e_i: float = -80.0
    tau_e: float = 5.0
    tau_i: float = 10.0
    v_threshold: float = -50.0
    v_reset: float = -65.0
    dt: float = 0.1

    def step(self, current: float, delta_ge: float = 0.0, delta_gi: float = 0.0) -> int:
        self.g_e += delta_ge
        self.g_i += delta_gi
        i_syn = self.g_e * (self.v - self.e_e) + self.g_i * (self.v - self.e_i)
        dv = (-self.g_l * (self.v - self.e_l) - i_syn + current) / self.c_m * self.dt
        self.v += dv
        self.g_e *= np.exp(-self.dt / self.tau_e)
        self.g_i *= np.exp(-self.dt / self.tau_i)
        if self.v >= self.v_threshold:
            self.v = self.v_reset
            return 1
        return 0

    def reset(self) -> None:
        self.v = self.e_l
        self.g_e = 0.0
        self.g_i = 0.0
