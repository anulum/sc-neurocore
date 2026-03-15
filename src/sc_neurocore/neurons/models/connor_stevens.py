# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class ConnorStevensNeuron:
    """Connor-Stevens 1977 — A-type potassium current, Type-I excitability.

    4 ODEs: v, m (Na activation), h (Na inactivation), n (K), a (A-type), b (A-type inactivation).
    """

    v: float = -68.0
    m: float = 0.01
    h: float = 0.99
    n: float = 0.1
    a: float = 0.5
    b: float = 0.1
    g_na: float = 120.0
    g_k: float = 20.0
    g_a: float = 47.7
    g_l: float = 0.3
    e_na: float = 55.0
    e_k: float = -72.0
    e_a: float = -75.0
    e_l: float = -17.0
    c_m: float = 1.0
    dt: float = 0.01
    v_threshold: float = 0.0

    def step(self, current: float) -> int:
        v_prev = self.v
        for _ in range(int(1.0 / max(self.dt, 0.001))):
            am = (
                0.38 * (self.v + 29.7) / (1.0 - np.exp(-(self.v + 29.7) / 10.0))
                if abs(self.v + 29.7) > 1e-6
                else 3.8
            )
            bm = 15.2 * np.exp(-(self.v + 54.7) / 18.0)
            ah = 0.266 * np.exp(-(self.v + 48.0) / 20.0)
            bh = 3.8 / (1.0 + np.exp(-(self.v + 18.0) / 10.0))
            an = (
                0.02 * (self.v + 45.7) / (1.0 - np.exp(-(self.v + 45.7) / 10.0))
                if abs(self.v + 45.7) > 1e-6
                else 0.2
            )
            bn = 0.25 * np.exp(-(self.v + 55.7) / 80.0)

            a_inf = (
                0.0761 * np.exp((self.v + 94.22) / 31.84) / (1.0 + np.exp((self.v + 1.17) / 28.93))
            ) ** (1.0 / 3.0)
            tau_a = 0.3632 + 1.158 / (1.0 + np.exp((self.v + 55.96) / 20.12))
            b_inf = 1.0 / (1.0 + np.exp((self.v + 53.3) / 14.54)) ** 4
            tau_b = 1.24 + 2.678 / (1.0 + np.exp((self.v + 50.0) / 16.027))

            self.m += (am * (1 - self.m) - bm * self.m) * self.dt
            self.h += (ah * (1 - self.h) - bh * self.h) * self.dt
            self.n += (an * (1 - self.n) - bn * self.n) * self.dt
            self.a += ((a_inf - self.a) / tau_a) * self.dt
            self.b += ((b_inf - self.b) / tau_b) * self.dt

            i_na = self.g_na * self.m**3 * self.h * (self.v - self.e_na)
            i_k = self.g_k * self.n**4 * (self.v - self.e_k)
            i_a = self.g_a * self.a**3 * self.b * (self.v - self.e_a)
            i_l = self.g_l * (self.v - self.e_l)

            self.v += (-i_na - i_k - i_a - i_l + current) / self.c_m * self.dt

        return 1 if (self.v >= self.v_threshold and v_prev < self.v_threshold) else 0

    def reset(self):
        self.v = -68.0
        self.m, self.h, self.n, self.a, self.b = 0.01, 0.99, 0.1, 0.5, 0.1
