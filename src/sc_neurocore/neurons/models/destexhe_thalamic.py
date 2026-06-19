# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Destexhe 1993 — thalamocortical relay with T-current and I_h

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DestexheThalamicNeuron:
    """Destexhe 1993 — thalamocortical relay with T-current and I_h.

    6 ODEs: V, m_Na, h_Na, n_K, m_T, h_T (+ optional h-current).

    Reference: Destexhe, A. et al. (1996). J. Comput. Neurosci. 3:19–46.
    """

    v: float = -65.0
    h_na: float = 0.6
    n_k: float = 0.3
    m_t: float = 0.0
    h_t: float = 1.0
    g_na: float = 100.0
    g_k: float = 10.0
    g_t: float = 2.0
    g_l: float = 0.05
    e_na: float = 50.0
    e_k: float = -90.0
    e_ca: float = 120.0
    e_l: float = -70.0
    dt: float = 0.02
    v_threshold: float = -20.0

    def step(self, current: float) -> int:
        v_prev = self.v
        for _ in range(5):
            m_na_inf = 1.0 / (1.0 + np.exp(-(self.v + 37.0) / 7.0))
            h_na_inf = 1.0 / (1.0 + np.exp((self.v + 41.0) / 4.0))
            n_k_inf = 1.0 / (1.0 + np.exp(-(self.v + 25.0) / 12.0))
            m_t_inf = 1.0 / (1.0 + np.exp(-(self.v + 57.0) / 6.5))
            h_t_inf = 1.0 / (1.0 + np.exp((self.v + 81.0) / 4.0))

            tau_h_na = 1.0 / (
                0.128 * np.exp(-(self.v + 46.0) / 18.0)
                + 4.0 / (1.0 + np.exp(-(self.v + 23.0) / 5.0))
            )
            tau_n_k = 1.0 / (0.032 * 5.0 + 0.5 * np.exp(-(self.v + 40.0) / 40.0)) if True else 1.0
            tau_h_t = (
                30.8
                + 211.4 * np.exp((self.v + 115.2) / 5.0) / (1.0 + np.exp((self.v + 86.0) / 3.2))
                if self.v < -81.0
                else 10.0
            )

            self.h_na += (h_na_inf - self.h_na) / max(tau_h_na, 0.1) * self.dt
            self.n_k += (n_k_inf - self.n_k) / max(tau_n_k, 0.1) * self.dt
            self.m_t = m_t_inf
            self.h_t += (h_t_inf - self.h_t) / max(tau_h_t, 0.1) * self.dt

            i_na = self.g_na * m_na_inf**3 * self.h_na * (self.v - self.e_na)
            i_k = self.g_k * self.n_k**4 * (self.v - self.e_k)
            i_t = self.g_t * self.m_t**2 * self.h_t * (self.v - self.e_ca)
            i_l = self.g_l * (self.v - self.e_l)
            self.v += (-i_na - i_k - i_t - i_l + current) * self.dt

        return 1 if (self.v >= self.v_threshold and v_prev < self.v_threshold) else 0

    def reset(self) -> None:
        self.v = -65.0
        self.h_na, self.n_k, self.m_t, self.h_t = 0.6, 0.3, 0.0, 1.0
