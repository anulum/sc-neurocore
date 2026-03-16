# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class ChayKeizerNeuron:
    """Chay & Keizer 1983 — pancreatic beta cell with Ca-dependent K.

    3 ODEs: V, n (delayed rectifier K), Ca (intracellular).
    I_Ca fast, I_K delayed rectifier, I_K(Ca) slow.
    """

    v: float = -50.0
    n: float = 0.01
    ca: float = 0.1  # uM
    g_ca: float = 20.0
    g_k: float = 25.0
    g_kca: float = 12.0
    g_l: float = 0.1
    e_ca: float = 100.0
    e_k: float = -75.0
    e_l: float = -40.0
    k_d: float = 1.0  # uM, Ca half-activation of K_Ca
    f_ca: float = 0.004  # Ca influx proportionality
    k_ca: float = 0.03  # Ca removal rate, ms^-1
    dt: float = 0.02
    v_threshold: float = -20.0

    def step(self, current: float) -> int:
        v_prev = self.v
        m_inf = 1.0 / (1.0 + np.exp(-(self.v + 25.0) / 8.0))
        n_inf = 1.0 / (1.0 + np.exp(-(self.v + 18.0) / 14.0))
        tau_n = 20.0 / (1.0 + np.exp((self.v + 18.0) / 14.0))
        q_kca = self.ca / (self.ca + self.k_d)

        i_ca = self.g_ca * m_inf * (self.v - self.e_ca)
        i_k = self.g_k * self.n * (self.v - self.e_k)
        i_kca = self.g_kca * q_kca * (self.v - self.e_k)
        i_l = self.g_l * (self.v - self.e_l)

        self.v += (-i_ca - i_k - i_kca - i_l + current) * self.dt
        self.n += (n_inf - self.n) / max(tau_n, 0.1) * self.dt
        self.ca = max(0.0, self.ca + (-self.f_ca * i_ca - self.k_ca * self.ca) * self.dt)

        return 1 if (self.v >= self.v_threshold and v_prev < self.v_threshold) else 0

    def reset(self):
        self.v, self.n, self.ca = -50.0, 0.01, 0.1
