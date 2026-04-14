# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — VIP Irregular-Spiking Interneuron

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class VIPNeuron:
    """VIP (vasoactive intestinal peptide) irregular-spiking interneuron.

    Na⁺, K⁺, A-type K⁺ (Kv4 for accommodation), leak. High input
    resistance, small soma. Disinhibitory role (inhibits SST+ and PV+).

    Reference: Porter et al. (1998); Bhatt et al. (2019).
    """

    v: float = -65.0
    h: float = 0.8
    n: float = 0.1
    a: float = 0.0
    b: float = 0.9
    g_na: float = 35.0
    g_k: float = 6.0
    g_a: float = 8.0
    g_l: float = 0.01
    e_na: float = 55.0
    e_k: float = -90.0
    e_l: float = -65.0
    c_m: float = 0.5
    dt: float = 0.025
    v_threshold: float = -20.0

    def step(self, current: float = 0.0) -> int:
        v_prev = self.v
        for _ in range(4):
            m_inf = 1.0 / (1.0 + math.exp(-(self.v + 30.0) / 9.5))
            h_inf = 1.0 / (1.0 + math.exp((self.v + 53.0) / 7.0))
            tau_h = 0.37 + 2.78 / (1.0 + math.exp((self.v + 40.5) / 6.0))
            self.h += (h_inf - self.h) / tau_h * self.dt

            n_inf = 1.0 / (1.0 + math.exp(-(self.v + 30.0) / 10.0))
            tau_n = 0.37 + 1.85 / (1.0 + math.exp((self.v + 27.0) / 15.0))
            self.n += (n_inf - self.n) / tau_n * self.dt

            a_inf = 1.0 / (1.0 + math.exp(-(self.v + 50.0) / 20.0))
            b_inf = 1.0 / (1.0 + math.exp((self.v + 78.0) / 6.0))
            self.a += (a_inf - self.a) / 5.0 * self.dt
            self.b += (b_inf - self.b) / 50.0 * self.dt

            i_na = self.g_na * m_inf**3 * self.h * (self.v - self.e_na)
            i_k = self.g_k * self.n**4 * (self.v - self.e_k)
            i_a = self.g_a * self.a**3 * self.b * (self.v - self.e_k)
            i_l = self.g_l * (self.v - self.e_l)

            self.v += (-i_na - i_k - i_a - i_l + current) / self.c_m * self.dt

        return 1 if self.v >= self.v_threshold and v_prev < self.v_threshold else 0

    def reset(self) -> None:
        self.v = -65.0
        self.h = 0.8
        self.n = 0.1
        self.a = 0.0
        self.b = 0.9
