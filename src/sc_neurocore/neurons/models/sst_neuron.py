# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SST+ Low-Threshold Spiking Interneuron

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class SSTNeuron:
    """SST+ (somatostatin) low-threshold spiking interneuron.

    Pospischil et al. 2008 LTS parameterisation with Na⁺, K⁺, M-current
    (Kv7 for adaptation), T-type Ca²⁺ (rebound bursting), Ih (sag), leak.

    Reference: Pospischil et al. (2008) Biol Cybern 99:427–441.
    """

    v: float = -65.0
    m: float = 0.02
    h: float = 0.8
    n: float = 0.2
    p: float = 0.0
    s: float = 0.9
    r: float = 0.1
    g_na: float = 50.0
    g_k: float = 5.0
    g_m: float = 0.12
    g_t: float = 0.01
    g_h: float = 0.02
    g_l: float = 0.05
    e_na: float = 50.0
    e_k: float = -90.0
    e_ca: float = 120.0
    e_h: float = -40.0
    e_l: float = -65.0
    c_m: float = 1.0
    dt: float = 0.025
    v_threshold: float = -20.0

    def step(self, current: float = 0.0) -> int:
        v_prev = self.v
        vt = -56.2
        for _ in range(4):
            dv = self.v - vt
            # Na+ gating (Pospischil)
            x_m = dv - 13.0
            alpha_m = 0.32 * 4.0 if abs(x_m) < 1e-6 else -0.32 * x_m / (math.exp(-x_m / 4.0) - 1.0)
            x_h = dv - 17.0
            beta_m = 0.28 * 5.0 if abs(x_h) < 1e-6 else 0.28 * x_h / (math.exp(x_h / 5.0) - 1.0)
            alpha_h = 0.128 * math.exp(-(dv - 17.0) / 18.0)
            beta_h = 4.0 / (1.0 + math.exp(-(dv - 40.0) / 5.0))
            # K+ gating
            x_n = dv - 15.0
            alpha_n = (
                0.032 * 5.0 if abs(x_n) < 1e-6 else -0.032 * x_n / (math.exp(-x_n / 5.0) - 1.0)
            )
            beta_n = 0.5 * math.exp(-(dv - 10.0) / 40.0)

            self.m += (alpha_m * (1.0 - self.m) - beta_m * self.m) * self.dt
            self.h += (alpha_h * (1.0 - self.h) - beta_h * self.h) * self.dt
            self.n += (alpha_n * (1.0 - self.n) - beta_n * self.n) * self.dt

            # M-current
            p_inf = 1.0 / (1.0 + math.exp(-(self.v + 35.0) / 10.0))
            tau_p = 400.0 / (
                3.3 * math.exp((self.v + 35.0) / 20.0) + math.exp(-(self.v + 35.0) / 20.0)
            )
            self.p += (p_inf - self.p) / tau_p * self.dt

            # T-type Ca²⁺
            m_t_inf = 1.0 / (1.0 + math.exp(-(self.v + 57.0) / 6.2))
            s_inf = 1.0 / (1.0 + math.exp((self.v + 81.0) / 4.0))
            tau_s = 30.0 + 200.0 / (1.0 + math.exp((self.v + 70.0) / 5.0))
            self.s += (s_inf - self.s) / tau_s * self.dt

            # Ih
            r_inf = 1.0 / (1.0 + math.exp((self.v + 80.0) / 10.0))
            tau_r = 100.0 + 500.0 / (
                math.exp(-(self.v + 70.0) / 20.0) + math.exp((self.v + 70.0) / 20.0)
            )
            self.r += (r_inf - self.r) / tau_r * self.dt

            i_na = self.g_na * self.m**3 * self.h * (self.v - self.e_na)
            i_k = self.g_k * self.n**4 * (self.v - self.e_k)
            i_m = self.g_m * self.p * (self.v - self.e_k)
            i_t = self.g_t * m_t_inf**2 * self.s * (self.v - self.e_ca)
            i_h = self.g_h * self.r * (self.v - self.e_h)
            i_l = self.g_l * (self.v - self.e_l)

            self.v += (-i_na - i_k - i_m - i_t - i_h - i_l + current) / self.c_m * self.dt

        return 1 if self.v >= self.v_threshold and v_prev < self.v_threshold else 0

    def reset(self) -> None:
        self.v = -65.0
        self.m = 0.02
        self.h = 0.8
        self.n = 0.2
        self.p = 0.0
        self.s = 0.9
        self.r = 0.1
