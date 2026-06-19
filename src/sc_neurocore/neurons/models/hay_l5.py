# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Hay et al. 2011 — Layer 5 thick-tufted pyramidal cell

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class HayL5PyramidalNeuron:
    """Hay et al. 2011 — Layer 5 thick-tufted pyramidal cell (reduced).

    Simplified 3-compartment reduction of the full morphological model.
    Compartments: soma (Na/K fast), apical trunk (Ca/Ih), apical tuft (Ca/NMDA).
    Reproduces BAC firing (backpropagation-activated calcium spike).

    Original: Hay, Hill, Schurmann, Markram & Segev, PLoS Comput Biol 7(7), 2011.

    Reference: Hay, E. et al. (2011). PLoS Comput. Biol. 7:e1002107.
    """

    # Soma
    v_s: float = -75.0
    h_na: float = 0.9
    n_k: float = 0.1
    # Apical trunk
    v_t: float = -75.0
    m_ca: float = 0.0
    h_ca: float = 1.0
    m_ih: float = 0.0
    # Apical tuft
    v_a: float = -75.0
    ca_a: float = 0.0001

    # Soma conductances
    g_na: float = 300.0
    g_k: float = 40.0
    g_l_s: float = 0.03
    e_na: float = 50.0
    e_k: float = -85.0
    e_l: float = -75.0

    # Trunk conductances
    g_ca_t: float = 2.0
    g_ih: float = 0.02
    g_l_t: float = 0.03
    e_ca: float = 140.0
    e_ih: float = -45.0

    # Tuft
    g_ca_a: float = 1.5
    g_kca: float = 2.5
    g_l_a: float = 0.03

    # Coupling
    g_st: float = 1.5
    g_ta: float = 0.8
    p_s: float = 0.15
    p_t: float = 0.25
    p_a: float = 0.60

    # Ca dynamics
    ca_decay: float = 200.0
    f_ca: float = 0.0002

    dt: float = 0.025
    v_threshold: float = -30.0
    c_m: float = 1.0

    def step(self, current_soma: float, current_tuft: float = 0.0) -> int:
        v_s_prev = self.v_s

        for _ in range(4):
            # Soma gating
            m_na_inf = 1.0 / (1.0 + np.exp(-(self.v_s + 38.0) / 7.0))
            h_na_inf = 1.0 / (1.0 + np.exp((self.v_s + 65.0) / 6.0))
            n_k_inf = 1.0 / (1.0 + np.exp(-(self.v_s + 25.0) / 12.0))
            tau_h = 0.5 + 14.0 / (1.0 + np.exp((self.v_s + 35.0) / 10.0))
            tau_n = 1.0 + 5.0 / (1.0 + np.exp((self.v_s + 30.0) / 10.0))

            self.h_na += (h_na_inf - self.h_na) / tau_h * self.dt
            self.n_k += (n_k_inf - self.n_k) / tau_n * self.dt

            i_na = self.g_na * m_na_inf**3 * self.h_na * (self.v_s - self.e_na)
            i_k = self.g_k * self.n_k**4 * (self.v_s - self.e_k)
            i_l_s = self.g_l_s * (self.v_s - self.e_l)
            i_st = self.g_st * (self.v_s - self.v_t) / self.p_s

            # Trunk gating
            m_ca_inf = 1.0 / (1.0 + np.exp(-(self.v_t + 27.0) / 7.0))
            h_ca_inf = 1.0 / (1.0 + np.exp((self.v_t + 52.0) / 5.0))
            m_ih_inf = 1.0 / (1.0 + np.exp((self.v_t + 75.0) / 5.5))
            tau_m_ca = 1.0
            tau_h_ca = 20.0
            tau_ih = 50.0

            self.m_ca += (m_ca_inf - self.m_ca) / tau_m_ca * self.dt
            self.h_ca += (h_ca_inf - self.h_ca) / tau_h_ca * self.dt
            self.m_ih += (m_ih_inf - self.m_ih) / tau_ih * self.dt

            i_ca_t = self.g_ca_t * self.m_ca**2 * self.h_ca * (self.v_t - self.e_ca)
            i_ih = self.g_ih * self.m_ih * (self.v_t - self.e_ih)
            i_l_t = self.g_l_t * (self.v_t - self.e_l)
            i_ts = self.g_st * (self.v_t - self.v_s) / self.p_t
            i_ta = self.g_ta * (self.v_t - self.v_a) / self.p_t

            # Tuft
            m_ca_a_inf = 1.0 / (1.0 + np.exp(-(self.v_a + 30.0) / 5.0))
            kca_act = self.ca_a / (self.ca_a + 0.001)
            i_ca_a = self.g_ca_a * m_ca_a_inf**2 * (self.v_a - self.e_ca)
            i_kca = self.g_kca * kca_act * (self.v_a - self.e_k)
            i_l_a = self.g_l_a * (self.v_a - self.e_l)
            i_at = self.g_ta * (self.v_a - self.v_t) / self.p_a

            # Update voltages
            self.v_s += (-i_na - i_k - i_l_s - i_st + current_soma / self.p_s) / self.c_m * self.dt
            self.v_t += (-i_ca_t - i_ih - i_l_t - i_ts - i_ta) / self.c_m * self.dt
            self.v_a += (
                (-i_ca_a - i_kca - i_l_a - i_at + current_tuft / self.p_a) / self.c_m * self.dt
            )

            # Ca dynamics in tuft
            self.ca_a = max(
                0.0, self.ca_a + (-self.f_ca * i_ca_a - self.ca_a / self.ca_decay) * self.dt
            )

        return 1 if (self.v_s >= self.v_threshold and v_s_prev < self.v_threshold) else 0

    def reset(self) -> None:
        self.v_s = self.v_t = self.v_a = -75.0
        self.h_na = 0.9
        self.n_k = 0.1
        self.m_ca = 0.0
        self.h_ca = 1.0
        self.m_ih = 0.0
        self.ca_a = 0.0001
