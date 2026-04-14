# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Cerebellar Basket Cell

from __future__ import annotations

import math
from dataclasses import dataclass


def _safe_rate(a: float, vhalf: float, v: float, k: float, fallback: float) -> float:
    d = v + vhalf
    if abs(d) < 1e-7:
        return fallback
    return a * d / (1.0 - math.exp(-d / k))


@dataclass
class CerebellarBasketNeuron:
    """Cerebellar basket cell — perisomatic-targeting interneuron.

    WB core + A-type K⁺ (transient outward) + Ca²⁺-dependent K⁺ (AHP).
    Distinct from cortical PV+ by A-current and pronounced AHP.

    Reference: Midtgaard (1992); Häusser & Clark (1997); WB (1996).
    """

    v: float = -65.0
    h: float = 0.8
    n: float = 0.1
    a: float = 0.0
    b: float = 0.9
    ca: float = 0.05
    g_na: float = 35.0
    g_k: float = 9.0
    g_a: float = 3.0
    g_kca: float = 2.0
    g_l: float = 0.1
    e_na: float = 55.0
    e_k: float = -90.0
    e_l: float = -65.0
    c_m: float = 1.0
    phi: float = 5.0
    dt: float = 0.01
    v_threshold: float = -20.0

    def step(self, current: float = 0.0) -> int:
        v_prev = self.v
        n_sub = max(1, int(0.5 / max(self.dt, 0.001)))
        for _ in range(n_sub):
            am = _safe_rate(0.1, 35.0, self.v, 10.0, 1.0)
            bm = 4.0 * math.exp(-(self.v + 60.0) / 18.0)
            m_inf = am / (am + bm)
            ah = 0.07 * math.exp(-(self.v + 58.0) / 20.0)
            bh = 1.0 / (1.0 + math.exp(-(self.v + 28.0) / 10.0))
            an = _safe_rate(0.01, 34.0, self.v, 10.0, 0.1)
            bn = 0.125 * math.exp(-(self.v + 44.0) / 80.0)

            self.h += self.phi * (ah * (1.0 - self.h) - bh * self.h) * self.dt
            self.n += self.phi * (an * (1.0 - self.n) - bn * self.n) * self.dt

            a_inf = 1.0 / (1.0 + math.exp(-(self.v + 45.0) / 15.0))
            b_inf = 1.0 / (1.0 + math.exp((self.v + 75.0) / 8.0))
            self.a += self.phi * (a_inf - self.a) / 5.0 * self.dt
            self.b += (b_inf - self.b) / 50.0 * self.dt

            q_inf = self.ca / (self.ca + 0.2)

            i_ca_entry = 0.01 * (self.v + 20.0) if self.v > -20.0 else 0.0
            self.ca += (-self.ca / 80.0 + i_ca_entry) * self.dt
            self.ca = max(0.0, self.ca)

            i_na = self.g_na * m_inf**3 * self.h * (self.v - self.e_na)
            i_k = self.g_k * self.n**4 * (self.v - self.e_k)
            i_a = self.g_a * self.a**3 * self.b * (self.v - self.e_k)
            i_kca = self.g_kca * q_inf * (self.v - self.e_k)
            i_l = self.g_l * (self.v - self.e_l)

            self.v += (-i_na - i_k - i_a - i_kca - i_l + current) / self.c_m * self.dt

        return 1 if self.v >= self.v_threshold and v_prev < self.v_threshold else 0

    def reset(self) -> None:
        self.v = -65.0
        self.h = 0.8
        self.n = 0.1
        self.a = 0.0
        self.b = 0.9
        self.ca = 0.05
