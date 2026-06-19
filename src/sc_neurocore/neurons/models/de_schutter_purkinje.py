# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — De Schutter & Bower 1994 — cerebellar Purkinje cell

"""Compact conductance-based Purkinje-cell model after De Schutter & Bower."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DeSchutterPurkinjeNeuron:
    """Single-compartment Purkinje-cell conductance model after De Schutter & Bower.

    The maintained Python model exposes five active gating variables:
    h_Na, n_K, m_CaP, h_CaP, and q_KCa. Use the audit index before treating this
    compact implementation as a full multi-compartment reconstruction.

    Reference: De Schutter, E. & Bower, J.M. (1994). J. Neurophysiol. 71:375–400.
    """

    v: float = -68.0
    h_na: float = 0.8
    n_k: float = 0.1
    m_cap: float = 0.0
    h_cap: float = 0.9
    q_kca: float = 0.0
    ca: float = 0.0001  # mM
    g_na: float = 125.0  # mS/cm^2
    g_k: float = 10.0
    g_cap: float = 45.0
    g_kca: float = 35.0
    g_l: float = 0.5
    e_na: float = 45.0
    e_k: float = -85.0
    e_ca: float = 135.0
    e_l: float = -68.0
    ca_decay: float = 0.02  # ms^-1
    f_ca: float = 0.00024  # mM/ms per mA/cm^2
    dt: float = 0.01
    v_threshold: float = -20.0

    def step(self, current: float) -> int:
        """Advance the compact conductance model and return a spike indicator."""
        v_prev = self.v
        for _ in range(5):
            m_na_inf = 1.0 / (1.0 + np.exp(-(self.v + 35.0) / 7.5))
            h_na_inf = 1.0 / (1.0 + np.exp((self.v + 55.0) / 7.0))
            n_k_inf = 1.0 / (1.0 + np.exp(-(self.v + 30.0) / 15.0))
            m_cap_inf = 1.0 / (1.0 + np.exp(-(self.v + 19.0) / 5.5))
            h_cap_inf = 1.0 / (1.0 + np.exp((self.v + 48.0) / 7.0))
            q_kca_inf = self.ca / (self.ca + 0.0002)

            tau_h_na = 0.5 + 14.0 / (1.0 + np.exp((self.v + 40.0) / 12.0))
            tau_n_k = 1.0 + 11.0 / (1.0 + np.exp((self.v + 15.0) / 8.0))
            tau_m_cap = 0.3
            tau_h_cap = 45.0
            tau_q = 1.0

            self.h_na += (h_na_inf - self.h_na) / tau_h_na * self.dt
            self.n_k += (n_k_inf - self.n_k) / tau_n_k * self.dt
            self.m_cap += (m_cap_inf - self.m_cap) / tau_m_cap * self.dt
            self.h_cap += (h_cap_inf - self.h_cap) / tau_h_cap * self.dt
            self.q_kca += (q_kca_inf - self.q_kca) / tau_q * self.dt

            i_na = self.g_na * m_na_inf**3 * self.h_na * (self.v - self.e_na)
            i_k = self.g_k * self.n_k**4 * (self.v - self.e_k)
            i_cap = self.g_cap * self.m_cap**2 * self.h_cap * (self.v - self.e_ca)
            i_kca = self.g_kca * self.q_kca * (self.v - self.e_k)
            i_l = self.g_l * (self.v - self.e_l)

            self.v += (-i_na - i_k - i_cap - i_kca - i_l + current) * self.dt
            self.ca = max(0.0, self.ca + (-self.f_ca * i_cap - self.ca_decay * self.ca) * self.dt)

        return 1 if (self.v >= self.v_threshold and v_prev < self.v_threshold) else 0

    def reset(self) -> None:
        """Restore voltage, gates, and calcium state to their defaults."""
        self.v = -68.0
        self.h_na, self.n_k, self.m_cap, self.h_cap, self.q_kca = 0.8, 0.1, 0.0, 0.9, 0.0
        self.ca = 0.0001
