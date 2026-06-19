# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Hill & Tononi 2005 — thalamocortical sleep/wake model

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class HillTononiNeuron:
    """Hill & Tononi 2005 — thalamocortical sleep/wake model.

    I_Na + I_K + I_h + I_T + I_KNa + I_L with Na-dependent K current.

    Reference: Hill, S. & Tononi, G. (2005). J. Neurophysiol. 93:1671–1698.
    """

    v: float = -65.0
    h_na: float = 0.6
    n_k: float = 0.3
    m_h: float = 0.0
    h_t: float = 0.9
    na_i: float = 5.0  # mM intracellular Na
    g_na: float = 50.0
    g_k: float = 5.0
    g_h: float = 1.0
    g_t: float = 3.0
    g_kna: float = 1.33
    g_l: float = 0.02
    e_na: float = 50.0
    e_k: float = -90.0
    e_h: float = -43.0
    e_ca: float = 120.0
    e_l: float = -70.0
    na_pump_max: float = 20.0  # mM/s, Na/K pump rate
    na_eq: float = 9.5  # mM, Na pump equilibrium
    dt: float = 0.05
    v_threshold: float = -20.0

    def step(self, current: float) -> int:
        v_prev = self.v
        m_na_inf = 1.0 / (1.0 + np.exp(-(self.v + 38.0) / 10.0))
        h_na_inf = 1.0 / (1.0 + np.exp((self.v + 43.0) / 6.0))
        n_k_inf = 1.0 / (1.0 + np.exp(-(self.v + 27.0) / 11.5))
        m_h_inf = 1.0 / (1.0 + np.exp((self.v + 75.0) / 5.5))
        m_t_inf = 1.0 / (1.0 + np.exp(-(self.v + 59.0) / 6.2))
        h_t_inf = 1.0 / (1.0 + np.exp((self.v + 83.0) / 4.0))
        w_kna = 0.37 / (1.0 + (38.7 / max(self.na_i, 0.01)) ** 3.5)

        tau_h_na = 1.0 + 10.0 / (1.0 + np.exp((self.v + 40.0) / 10.0))
        tau_n_k = 5.0 + 47.0 * np.exp(-(((self.v + 50.0) / 25.0) ** 2))
        tau_m_h = 20.0 + 1000.0 / (np.exp((self.v + 71.5) / 14.2) + np.exp(-(self.v + 89.0) / 11.6))
        tau_h_t = (
            30.8 + 211.4 * np.exp((self.v + 115.2) / 5.0) / (1.0 + np.exp((self.v + 86.0) / 3.2))
            if self.v < -81.0
            else 10.0
        )

        self.h_na += (h_na_inf - self.h_na) / max(tau_h_na, 0.1) * self.dt
        self.n_k += (n_k_inf - self.n_k) / max(tau_n_k, 0.1) * self.dt
        self.m_h += (m_h_inf - self.m_h) / max(tau_m_h, 1.0) * self.dt
        self.h_t += (h_t_inf - self.h_t) / max(tau_h_t, 0.1) * self.dt

        i_na = self.g_na * m_na_inf**3 * self.h_na * (self.v - self.e_na)
        i_k = self.g_k * self.n_k**4 * (self.v - self.e_k)
        i_h = self.g_h * self.m_h * (self.v - self.e_h)
        i_t = self.g_t * m_t_inf**2 * self.h_t * (self.v - self.e_ca)
        i_kna = self.g_kna * w_kna * (self.v - self.e_k)
        i_l = self.g_l * (self.v - self.e_l)

        self.v += (-i_na - i_k - i_h - i_t - i_kna - i_l + current) * self.dt
        # Hill & Tononi 2005, Eq. 8: Na accumulation from i_na, pump removal
        self.na_i += (
            -0.001 * i_na - self.na_pump_max * (self.na_i / (self.na_i + self.na_eq))
        ) * self.dt
        self.na_i = max(self.na_i, 0.0)

        return 1 if (self.v >= self.v_threshold and v_prev < self.v_threshold) else 0

    def reset(self) -> None:
        self.v, self.h_na, self.n_k, self.m_h, self.h_t = -65.0, 0.6, 0.3, 0.0, 0.9
        self.na_i = 5.0
