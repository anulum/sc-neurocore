# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Marder & Selverston 1992 — stomatogastric ganglion neuron

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MarderSTGNeuron:
    """Marder & Selverston 1992 — stomatogastric ganglion neuron.

    7 currents: I_Na, I_CaT, I_CaS, I_A, I_KCa, I_Kd, I_H, I_L.
    LP-like model from the pyloric CPG.

    Reference: Marder, E. & Calabrese, R.L. (1996). Physiol. Rev. 76:687–717.
    """

    v: float = -60.0
    m_na: float = 0.0
    h_na: float = 0.9
    m_cat: float = 0.0
    h_cat: float = 0.9
    m_cas: float = 0.0
    m_a: float = 0.0
    h_a: float = 0.9
    m_kd: float = 0.0
    m_h: float = 0.0
    ca: float = 0.05  # uM
    g_na: float = 200.0
    g_cat: float = 2.5
    g_cas: float = 4.0
    g_a: float = 50.0
    g_kca: float = 25.0
    g_kd: float = 75.0
    g_h: float = 0.01
    g_l: float = 0.01
    e_na: float = 50.0
    e_ca: float = 80.0
    e_k: float = -80.0
    e_h: float = -20.0
    e_l: float = -50.0
    ca_decay: float = 0.02
    f_ca: float = 0.0003
    dt: float = 0.05
    v_threshold: float = -20.0

    def _boltz(self, v: float, v_half: float, k: float) -> float:
        return float(1.0 / (1.0 + np.exp((v_half - v) / k)))

    def step(self, current: float) -> int:
        v_prev = self.v
        m_na_inf = self._boltz(self.v, -25.5, 5.29)
        h_na_inf = self._boltz(self.v, -48.9, -5.18)
        m_cat_inf = self._boltz(self.v, -27.1, 7.2)
        h_cat_inf = self._boltz(self.v, -32.1, -5.5)
        m_cas_inf = self._boltz(self.v, -33.0, 8.1)
        m_a_inf = self._boltz(self.v, -27.2, 8.7)
        h_a_inf = self._boltz(self.v, -56.9, -4.9)
        m_kd_inf = self._boltz(self.v, -12.3, 11.8)
        m_h_inf = self._boltz(self.v, -70.0, -6.0)

        self.m_na = m_na_inf
        self.h_na += (h_na_inf - self.h_na) / 1.5 * self.dt
        self.m_cat += (m_cat_inf - self.m_cat) / 7.2 * self.dt
        self.h_cat += (h_cat_inf - self.h_cat) / 55.0 * self.dt
        self.m_cas += (m_cas_inf - self.m_cas) / 14.0 * self.dt
        self.m_a += (m_a_inf - self.m_a) / 11.6 * self.dt
        self.h_a += (h_a_inf - self.h_a) / 38.6 * self.dt
        self.m_kd += (m_kd_inf - self.m_kd) / 7.2 * self.dt
        self.m_h += (m_h_inf - self.m_h) / 272.0 * self.dt

        kca_act = self.ca / (self.ca + 3.0)
        i_na = self.g_na * self.m_na**3 * self.h_na * (self.v - self.e_na)
        i_cat = self.g_cat * self.m_cat**3 * self.h_cat * (self.v - self.e_ca)
        i_cas = self.g_cas * self.m_cas**3 * (self.v - self.e_ca)
        i_a = self.g_a * self.m_a**3 * self.h_a * (self.v - self.e_k)
        i_kca = self.g_kca * kca_act**4 * (self.v - self.e_k)
        i_kd = self.g_kd * self.m_kd**4 * (self.v - self.e_k)
        i_h = self.g_h * self.m_h * (self.v - self.e_h)
        i_l = self.g_l * (self.v - self.e_l)

        i_total = -i_na - i_cat - i_cas - i_a - i_kca - i_kd - i_h - i_l + current
        self.v += i_total * self.dt
        i_ca_total = i_cat + i_cas
        self.ca = max(0.0, self.ca + (-self.f_ca * i_ca_total - self.ca_decay * self.ca) * self.dt)

        return 1 if (self.v >= self.v_threshold and v_prev < self.v_threshold) else 0

    def reset(self) -> None:
        self.v = -60.0
        self.m_na, self.h_na = 0.0, 0.9
        self.m_cat, self.h_cat = 0.0, 0.9
        self.m_cas = 0.0
        self.m_a, self.h_a = 0.0, 0.9
        self.m_kd, self.m_h = 0.0, 0.0
        self.ca = 0.05
