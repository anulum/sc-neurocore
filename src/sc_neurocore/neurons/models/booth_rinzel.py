# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Booth & Rinzel 1995 — bistable motoneuron, 2-compartment

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class BoothRinzelNeuron:
    """Booth & Rinzel 1995 — bistable motoneuron, 2-compartment.

    C dVs/dt = -I_Na(Vs) - I_K(Vs) - I_L(Vs) - gc*(Vs - Vd)/p + I/p
    C dVd/dt = -I_Ca(Vd) - I_KCa(Vd) - I_L(Vd) - gc*(Vd - Vs)/(1-p)
    dq/dt   = (q_inf(Vd) - q) / tau_q
    dCa/dt  = -f * (alpha_Ca * I_Ca + k_Ca * Ca)
    """

    vs: float = -65.0
    vd: float = -65.0
    h: float = 0.9
    n: float = 0.0
    q: float = 0.0
    ca: float = 0.0
    p: float = 0.5
    gc: float = 0.1
    g_na: float = 120.0
    g_k: float = 20.0
    g_ca: float = 14.0
    g_kca: float = 5.0
    g_l: float = 0.51
    e_na: float = 55.0
    e_k: float = -80.0
    e_ca: float = 80.0
    e_l: float = -60.0
    c_m: float = 1.0
    alpha_ca: float = 0.009
    k_ca: float = 0.18
    f_ca: float = 0.0025
    dt: float = 0.025
    v_threshold: float = -20.0

    def step(self, current: float) -> int:
        vs_prev = self.vs
        for _ in range(4):
            # Soma: fast Na + delayed-rectifier K
            m_inf = 1.0 / (1.0 + np.exp(-(self.vs + 35.0) / 7.8))
            h_inf = 1.0 / (1.0 + np.exp((self.vs + 55.0) / 7.0))
            tau_h = 30.0 / (np.exp((self.vs + 50.0) / 15.0) + np.exp(-(self.vs + 50.0) / 16.0))
            n_inf = 1.0 / (1.0 + np.exp(-(self.vs + 28.0) / 15.0))
            tau_n = 7.0 / (np.exp((self.vs + 40.0) / 40.0) + np.exp(-(self.vs + 40.0) / 50.0))

            self.h += (h_inf - self.h) / tau_h * self.dt
            self.n += (n_inf - self.n) / tau_n * self.dt

            i_na = self.g_na * m_inf**3 * self.h * (self.vs - self.e_na)
            i_k = self.g_k * self.n**4 * (self.vs - self.e_k)
            i_ls = self.g_l * (self.vs - self.e_l)
            i_coup_s = self.gc * (self.vs - self.vd) / self.p

            dvs = (-i_na - i_k - i_ls - i_coup_s + current / self.p) / self.c_m * self.dt

            # Dendrite: Ca + KCa
            s_inf = 1.0 / (1.0 + np.exp(-(self.vd + 22.0) / 5.0))
            q_inf = 1.0 / (1.0 + np.exp(-(self.vd + 35.0) / 2.0))
            tau_q = 400.0

            self.q += (q_inf - self.q) / tau_q * self.dt

            i_ca = self.g_ca * s_inf**2 * (self.vd - self.e_ca)
            chi = min(self.ca / 250.0, 1.0)
            i_kca = self.g_kca * chi * (self.vd - self.e_k)
            i_ld = self.g_l * (self.vd - self.e_l)
            i_coup_d = self.gc * (self.vd - self.vs) / (1.0 - self.p)

            dvd = (-i_ca - i_kca - i_ld - i_coup_d) / self.c_m * self.dt
            self.ca += self.f_ca * (-self.alpha_ca * i_ca - self.k_ca * self.ca) * self.dt
            self.ca = max(self.ca, 0.0)

            self.vs += dvs
            self.vd += dvd

        return 1 if (self.vs >= self.v_threshold and vs_prev < self.v_threshold) else 0

    def reset(self):
        self.vs = -65.0
        self.vd = -65.0
        self.h, self.n, self.q = 0.9, 0.0, 0.0
        self.ca = 0.0
