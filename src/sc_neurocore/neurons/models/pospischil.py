# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Pospischil et al. 2008 — minimal HH for 5 cortical cell

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PospischilNeuron:
    """Pospischil et al. 2008 — minimal HH for 5 cortical cell types.

    C dV/dt = -I_Na - I_Kd - I_M - I_L + I_ext
    I_M (slow K+) provides adaptation; its conductance distinguishes:
      RS (g_m=0.07), FS (g_m=0), IB (g_m=0.03), LTS (g_m=0.03+I_T).
    Default parameters: RS cortical pyramidal.

    Reference: Pospischil, M. et al. (2008). Biol. Cybern. 99:427–441.
    """

    v: float = -70.0
    m: float = 0.05
    h: float = 0.6
    n: float = 0.3
    p: float = 0.0
    g_na: float = 50.0
    g_kd: float = 5.0
    g_m: float = 0.07
    g_l: float = 0.1
    e_na: float = 50.0
    e_k: float = -90.0
    e_l: float = -70.0
    c_m: float = 1.0
    vt: float = -56.2
    dt: float = 0.025
    v_threshold: float = -20.0

    def step(self, current: float) -> int:
        v_prev = self.v
        for _ in range(4):
            dv = self.v - self.vt
            am = -0.32 * (dv - 13.0) / (np.exp(-(dv - 13.0) / 4.0) - 1.0 + 1e-12)
            bm = 0.28 * (dv - 40.0) / (np.exp((dv - 40.0) / 5.0) - 1.0 + 1e-12)
            ah = 0.128 * np.exp(-(dv - 17.0) / 18.0)
            bh = 4.0 / (1.0 + np.exp(-(dv - 40.0) / 5.0))
            an = -0.032 * (dv - 15.0) / (np.exp(-(dv - 15.0) / 5.0) - 1.0 + 1e-12)
            bn = 0.5 * np.exp(-(dv - 10.0) / 40.0)
            p_inf = 1.0 / (1.0 + np.exp(-(self.v + 35.0) / 10.0))
            tau_p = 608.0 / (3.3 * np.exp((self.v + 35.0) / 20.0) + np.exp(-(self.v + 35.0) / 20.0))

            self.m += (am * (1 - self.m) - bm * self.m) * self.dt
            self.h += (ah * (1 - self.h) - bh * self.h) * self.dt
            self.n += (an * (1 - self.n) - bn * self.n) * self.dt
            self.p += (p_inf - self.p) / tau_p * self.dt

            i_na = self.g_na * self.m**3 * self.h * (self.v - self.e_na)
            i_kd = self.g_kd * self.n**4 * (self.v - self.e_k)
            i_m = self.g_m * self.p * (self.v - self.e_k)
            i_l = self.g_l * (self.v - self.e_l)

            self.v += (-i_na - i_kd - i_m - i_l + current) / self.c_m * self.dt

        return 1 if (self.v >= self.v_threshold and v_prev < self.v_threshold) else 0

    def reset(self) -> None:
        self.v = -70.0
        self.m, self.h, self.n, self.p = 0.05, 0.6, 0.3, 0.0
