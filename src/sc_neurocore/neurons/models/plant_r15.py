# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Plant 1981 — Aplysia R15 parabolic burster, 5 ODEs

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PlantR15Neuron:
    """Plant 1981 — Aplysia R15 parabolic burster, 5 ODEs.

    dV/dt  = (-I_Na - I_K - I_Ca - I_L - I_KCa + I_ext) / C
    dm/dt  = (m_inf(V) - m) / tau_m(V)
    dh/dt  = (h_inf(V) - h) / tau_h(V)
    dn/dt  = (n_inf(V) - n) / tau_n(V)
    dCa/dt = -k_Ca * I_Ca - Ca / tau_Ca

    Reference: Plant, R.E. & Kim, M. (1976). Biophys. J. 16:227–244.
    """

    v: float = -50.0
    m: float = 0.05
    h: float = 0.6
    n: float = 0.3
    ca: float = 0.1
    g_na: float = 4.0
    g_k: float = 0.3
    g_ca: float = 0.004
    g_l: float = 0.003
    g_kca: float = 0.03
    e_na: float = 30.0
    e_k: float = -75.0
    e_ca: float = 140.0
    e_l: float = -40.0
    c_m: float = 1.0
    k_ca: float = 0.0085
    tau_ca: float = 500.0
    dt: float = 0.05
    v_threshold: float = -10.0

    def step(self, current: float) -> int:
        v_prev = self.v
        for _ in range(5):
            am = 0.1 * (50.0 + self.v) / (1.0 - np.exp(-(50.0 + self.v) / 10.0) + 1e-12)
            bm = 4.0 * np.exp(-(75.0 + self.v) / 18.0)
            ah = 0.07 * np.exp(-(self.v + 50.0) / 20.0)
            bh = 1.0 / (1.0 + np.exp(-(20.0 + self.v) / 10.0))
            an = 0.01 * (55.0 + self.v) / (1.0 - np.exp(-(55.0 + self.v) / 10.0) + 1e-12)
            bn = 0.125 * np.exp(-(65.0 + self.v) / 80.0)

            self.m += (am * (1 - self.m) - bm * self.m) * self.dt
            self.h += (ah * (1 - self.h) - bh * self.h) * self.dt
            self.n += (an * (1 - self.n) - bn * self.n) * self.dt

            m_ca_inf = 1.0 / (1.0 + np.exp(-(self.v + 25.0) / 5.0))
            i_na = self.g_na * self.m**3 * self.h * (self.v - self.e_na)
            i_k = self.g_k * self.n**4 * (self.v - self.e_k)
            i_ca = self.g_ca * m_ca_inf**2 * (self.v - self.e_ca)
            i_kca = self.g_kca * self.ca / (0.5 + self.ca) * (self.v - self.e_k)
            i_l = self.g_l * (self.v - self.e_l)

            self.v += (-i_na - i_k - i_ca - i_kca - i_l + current) / self.c_m * self.dt
            self.ca += (-self.k_ca * i_ca - self.ca / self.tau_ca) * self.dt
            self.ca = max(self.ca, 0.0)

        return 1 if (self.v >= self.v_threshold and v_prev < self.v_threshold) else 0

    def reset(self) -> None:
        self.v = -50.0
        self.m, self.h, self.n = 0.05, 0.6, 0.3
        self.ca = 0.1
