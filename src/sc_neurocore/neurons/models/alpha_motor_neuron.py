# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Alpha Motor Neuron

from __future__ import annotations

import math
from dataclasses import dataclass


def _safe_rate(a: float, vhalf: float, v: float, k: float, fallback: float) -> float:
    d = v + vhalf
    if abs(d) < 1e-7:
        return fallback
    return a * d / (1.0 - math.exp(-d / k))


@dataclass
class AlphaMotorNeuron:
    """Alpha motor neuron — spinal cord, innervates extrafusal muscle fibres.

    WB Na⁺/K⁺ core + persistent inward current (PIC, L-type Ca²⁺ for
    plateau potentials) + Ca²⁺-dependent AHP (SK channels for rate
    limiting). Larger soma (C_m=1.5).

    Reference: Powers & Binder (2001) J Neurophysiol 86;
    Heckman & Enoka (2012) Compr Physiol 2(4).
    """

    v: float = -65.0
    h: float = 0.8
    n: float = 0.1
    m_pic: float = 0.0
    h_pic: float = 1.0
    ca: float = 0.0
    ca_buf: float = 0.0
    g_na: float = 35.0
    g_k: float = 9.0
    g_pic: float = 0.15
    g_ahp: float = 3.0
    g_l: float = 0.3
    e_na: float = 55.0
    e_k: float = -90.0
    e_ca: float = 120.0
    e_l: float = -65.0
    c_m: float = 1.5
    phi: float = 4.0
    tau_ca: float = 150.0
    buf_ratio: float = 0.003
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

            m_pic_inf = 1.0 / (1.0 + math.exp(-(self.v + 40.0) / 5.0))
            self.m_pic += (m_pic_inf - self.m_pic) / 50.0 * self.dt

            h_pic_inf = 1.0 / (1.0 + math.exp((self.v + 40.0) / 8.0))
            tau_h_pic = 200.0 + 100.0 / max(0.01, 1.0 + ((self.v + 40.0) / 10.0) ** 2)
            self.h_pic += (h_pic_inf - self.h_pic) / tau_h_pic * self.dt
            self.h_pic = max(0.0, min(1.0, self.h_pic))

            i_ca_entry = self.g_pic * self.m_pic * self.h_pic * (self.v - self.e_ca)
            ca_influx = -i_ca_entry * 0.001 if i_ca_entry < 0.0 else 0.0
            ca_spike = 0.02 if self.v > -10.0 else 0.0
            free_ca_change = (ca_influx + ca_spike) * self.buf_ratio
            self.ca += (-self.ca / self.tau_ca + free_ca_change) * self.dt
            self.ca = max(0.0, self.ca)

            self.ca_buf += (
                (ca_influx + ca_spike) * (1.0 - self.buf_ratio)
                - self.ca_buf / (self.tau_ca * 5.0)
            ) * self.dt
            self.ca_buf = max(0.0, self.ca_buf)

            ca_total = self.ca + self.ca_buf * 0.01
            ahp_inf = ca_total ** 2 / (ca_total ** 2 + 0.25)

            i_na = self.g_na * m_inf**3 * self.h * (self.v - self.e_na)
            i_k = self.g_k * self.n**4 * (self.v - self.e_k)
            i_pic = self.g_pic * self.m_pic * self.h_pic * (self.v - self.e_ca)
            i_ahp = self.g_ahp * ahp_inf * (self.v - self.e_k)
            i_l = self.g_l * (self.v - self.e_l)

            self.v += (-i_na - i_k - i_pic - i_ahp - i_l + current) / self.c_m * self.dt

        return 1 if self.v >= self.v_threshold and v_prev < self.v_threshold else 0

    def reset(self) -> None:
        self.v = -65.0
        self.h = 0.8
        self.n = 0.1
        self.m_pic = 0.0
        self.h_pic = 1.0
        self.ca = 0.0
        self.ca_buf = 0.0
