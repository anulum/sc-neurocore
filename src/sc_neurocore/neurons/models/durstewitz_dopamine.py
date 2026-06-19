# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Durstewitz, Seamans & Sejnowski 2000 — PFC neuron with

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DurstewitzDopamineNeuron:
    """Durstewitz, Seamans & Sejnowski 2000 — PFC neuron with D1 modulation.

    D1 agonism enhances NMDA (g_nmda_scale), reduces Na window current
    (v_shift_na), and enhances K (g_k_scale), stabilising up-states.

    Reference: Durstewitz, D. et al. (2000). J. Neurophysiol. 83:1733–1750.
    """

    v: float = -65.0
    h_na: float = 0.7
    n_k: float = 0.2
    g_na: float = 45.0
    g_k: float = 18.0
    g_nmda: float = 0.5
    g_l: float = 0.02
    e_na: float = 55.0
    e_k: float = -80.0
    e_nmda: float = 0.0
    e_l: float = -65.0
    mg: float = 1.0  # mM extracellular Mg2+
    d1_level: float = 0.0  # 0..1 D1 agonism
    g_nmda_scale: float = 2.5  # max NMDA boost at d1=1
    g_k_scale: float = 1.5
    v_shift_na: float = -5.0  # mV, Na half-act shift at d1=1
    dt: float = 0.05
    v_threshold: float = -20.0

    def step(self, current: float) -> int:
        v_prev = self.v
        d = self.d1_level
        v_sh = d * self.v_shift_na

        m_na_inf = 1.0 / (1.0 + np.exp(-(self.v + 30.0 + v_sh) / 9.5))
        h_na_inf = 1.0 / (1.0 + np.exp((self.v + 53.0) / 7.0))
        n_k_inf = 1.0 / (1.0 + np.exp(-(self.v + 30.0) / 10.0))
        tau_h = 0.5 + 14.0 / (1.0 + np.exp((self.v + 50.0) / 12.0))
        tau_n = 1.0 + 11.0 / (1.0 + np.exp((self.v + 40.0) / 10.0))

        self.h_na += (h_na_inf - self.h_na) / tau_h * self.dt
        self.n_k += (n_k_inf - self.n_k) / tau_n * self.dt

        # Jahr & Stevens 1990, Mg block
        mg_block = 1.0 / (1.0 + self.mg / 3.57 * np.exp(-0.062 * self.v))
        nmda_g = self.g_nmda * (1.0 + d * (self.g_nmda_scale - 1.0))
        k_g = self.g_k * (1.0 + d * (self.g_k_scale - 1.0))

        i_na = self.g_na * m_na_inf**3 * self.h_na * (self.v - self.e_na)
        i_k = k_g * self.n_k**4 * (self.v - self.e_k)
        i_nmda = nmda_g * mg_block * (self.v - self.e_nmda)
        i_l = self.g_l * (self.v - self.e_l)

        self.v += (-i_na - i_k - i_nmda - i_l + current) * self.dt
        return 1 if (self.v >= self.v_threshold and v_prev < self.v_threshold) else 0

    def reset(self) -> None:
        self.v, self.h_na, self.n_k = -65.0, 0.7, 0.2
