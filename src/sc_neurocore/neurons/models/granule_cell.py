# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Cerebellar Granule Cell

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class GranuleCell:
    """Cerebellar granule cell — D'Angelo et al. 2001 model.

    Smallest and most numerous neuron in the brain. 7 ionic currents:
    INa (m³h), IKdr (n⁴), IKA (a³b), ICaT (mT²s), IKCa (Hill), Ih (r),
    plus tonic GABA. Ca²⁺ dynamics with KCa half-saturation.

    Reference: D'Angelo et al. (2001) J Neurosci 21:759–770.
    """

    v: float = -70.0
    m: float = 0.02
    h: float = 0.85
    n: float = 0.05
    a: float = 0.1
    b: float = 0.8
    m_t: float = 0.01
    s: float = 0.95
    ca: float = 0.05
    r: float = 0.1
    c_m: float = 1.0
    g_na: float = 17.0
    g_kdr: float = 9.0
    g_ka: float = 1.0
    g_t: float = 0.5
    g_kca: float = 3.5
    g_h: float = 0.03
    g_l: float = 0.1
    g_tonic: float = 0.2
    e_na: float = 87.4
    e_k: float = -84.7
    e_ca: float = 129.3
    e_h: float = -40.0
    e_l: float = -58.0
    e_gaba: float = -75.0
    tau_ca: float = 10.0
    kd_kca: float = 0.2
    dt: float = 0.5
    sub_steps: int = 4
    gain: float = 1.0

    @staticmethod
    def _boltz(v: float, vh: float, k: float) -> float:
        return 1.0 / (1.0 + math.exp(-(v - vh) / k))

    def step(self, current: float = 0.0) -> int:
        inp = self.gain * current
        dt_sub = self.dt / self.sub_steps
        v_prev = self.v

        for _ in range(self.sub_steps):
            v = self.v
            bz = self._boltz

            m_inf = bz(v, -30.0, 7.0)
            tau_m = 0.1 + 0.3 / max(0.01, 1.0 + ((v + 30.0) / 10.0) ** 2)
            self.m += dt_sub * (m_inf - self.m) / tau_m

            h_inf = bz(v, -52.0, -6.0)
            tau_h = 0.5 + 5.0 / max(0.01, 1.0 + ((v + 50.0) / 15.0) ** 2)
            self.h += dt_sub * (h_inf - self.h) / tau_h

            n_inf = bz(v, -35.0, 8.0)
            tau_n = 1.0 + 5.0 / max(0.01, 1.0 + ((v + 35.0) / 15.0) ** 2)
            self.n += dt_sub * (n_inf - self.n) / tau_n

            a_inf = bz(v, -50.0, 20.0)
            self.a += dt_sub * (a_inf - self.a) / 2.0

            b_inf = bz(v, -70.0, -6.0)
            self.b += dt_sub * (b_inf - self.b) / 50.0

            mt_inf = bz(v, -52.0, 5.0)
            self.m_t += dt_sub * (mt_inf - self.m_t) / 1.0

            s_inf = bz(v, -60.0, -6.5)
            tau_s = 20.0 + 50.0 / max(0.01, 1.0 + ((v + 65.0) / 10.0) ** 2)
            self.s += dt_sub * (s_inf - self.s) / tau_s

            r_inf = bz(v, -80.0, -10.0)
            tau_r = 50.0 + 200.0 / max(0.01, 1.0 + ((v + 80.0) / 20.0) ** 2)
            self.r += dt_sub * (r_inf - self.r) / tau_r

            for attr in ('m', 'h', 'n', 'a', 'b', 'm_t', 's', 'r'):
                setattr(self, attr, max(0.0, min(1.0, getattr(self, attr))))

            i_ca_t = self.g_t * self.m_t ** 2 * self.s * (v - self.e_ca)
            ca_entry = -i_ca_t * 0.001 if i_ca_t < 0.0 else 0.0
            self.ca += dt_sub * (-self.ca / self.tau_ca + ca_entry)
            self.ca = max(0.0, self.ca)

            kca_inf = self.ca ** 2 / (self.ca ** 2 + self.kd_kca ** 2)

            i_na = self.g_na * self.m ** 3 * self.h * (v - self.e_na)
            i_kdr = self.g_kdr * self.n ** 4 * (v - self.e_k)
            i_ka = self.g_ka * self.a ** 3 * self.b * (v - self.e_k)
            i_kca = self.g_kca * kca_inf * (v - self.e_k)
            i_h = self.g_h * self.r * (v - self.e_h)
            i_l = self.g_l * (v - self.e_l)
            i_gaba = self.g_tonic * (v - self.e_gaba)

            dv_val = (-(i_na + i_kdr + i_ka + i_ca_t + i_kca + i_h + i_l + i_gaba) + inp) / self.c_m
            self.v += dt_sub * dv_val

        self.v = max(-100.0, min(60.0, self.v))
        if not math.isfinite(self.v):
            self.v = -70.0

        return 1 if self.v >= 0.0 and v_prev < 0.0 else 0

    def reset(self) -> None:
        self.v = -70.0
        self.m = 0.02
        self.h = 0.85
        self.n = 0.05
        self.a = 0.1
        self.b = 0.8
        self.m_t = 0.01
        self.s = 0.95
        self.ca = 0.05
        self.r = 0.1
