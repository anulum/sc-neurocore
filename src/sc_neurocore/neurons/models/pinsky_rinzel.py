# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class PinskyRinzelNeuron:
    """Pinsky-Rinzel 1994 — 2-compartment pyramidal cell.

    Soma (fast Na/K) coupled to dendrite (Ca/KAHP) via gc.
    Minimal model for burst generation in cortical pyramidal cells.
    """

    v_s: float = -60.0
    v_d: float = -60.0
    h: float = 0.9
    n: float = 0.1
    s: float = 0.0
    c: float = 0.0
    q: float = 0.0
    gc: float = 2.1
    p: float = 0.5
    g_na: float = 30.0
    g_kdr: float = 15.0
    g_ca: float = 10.0
    g_kahp: float = 0.8
    g_l: float = 0.1
    e_na: float = 60.0
    e_k: float = -75.0
    e_ca: float = 80.0
    e_l: float = -60.0
    dt: float = 0.02
    v_threshold: float = -20.0

    def step(self, current_soma: float, current_dend: float = 0.0) -> int:
        v_prev = self.v_s
        am = (
            0.32 * (self.v_s + 54.0) / (1.0 - np.exp(-(self.v_s + 54.0) / 4.0))
            if abs(self.v_s + 54.0) > 1e-6
            else 8.0
        )
        bm = (
            0.28 * (self.v_s + 27.0) / (np.exp((self.v_s + 27.0) / 5.0) - 1.0)
            if abs(self.v_s + 27.0) > 1e-6
            else 5.6
        )
        m_inf = am / (am + bm)

        ah = 0.128 * np.exp(-(self.v_s + 50.0) / 18.0)
        bh = 4.0 / (1.0 + np.exp(-(self.v_s + 27.0) / 5.0))
        an = (
            0.032 * (self.v_s + 52.0) / (1.0 - np.exp(-(self.v_s + 52.0) / 5.0))
            if abs(self.v_s + 52.0) > 1e-6
            else 0.32
        )
        bn = 0.5 * np.exp(-(self.v_s + 57.0) / 40.0)

        s_inf = 1.0 / (1.0 + np.exp(-(self.v_d + 20.0) / 9.0))
        c_inf = min(self.c, 1.0) if self.c > 0 else 0.0

        # Soma
        i_na = self.g_na * m_inf**2 * self.h * (self.v_s - self.e_na)
        i_kdr = self.g_kdr * self.n**2 * (self.v_s - self.e_k)
        i_ls = self.g_l * (self.v_s - self.e_l)
        i_ds = (self.gc / self.p) * (self.v_s - self.v_d)

        # Dendrite
        i_ca = self.g_ca * self.s**2 * (self.v_d - self.e_ca)
        i_kahp = self.g_kahp * self.q * (self.v_d - self.e_k)
        i_ld = self.g_l * (self.v_d - self.e_l)
        i_sd = (self.gc / (1 - self.p)) * (self.v_d - self.v_s)

        self.v_s += (-i_na - i_kdr - i_ls - i_ds + current_soma / self.p) * self.dt
        self.v_d += (-i_ca - i_kahp - i_ld - i_sd + current_dend / (1 - self.p)) * self.dt
        self.h += (ah * (1 - self.h) - bh * self.h) * self.dt
        self.n += (an * (1 - self.n) - bn * self.n) * self.dt
        self.s += ((s_inf - self.s) / 5.0) * self.dt
        self.c = max(0.0, self.c + (-0.13 * i_ca - 0.075 * self.c) * self.dt)
        q_inf = min(self.c / (self.c + 2.0), 1.0)
        self.q += ((q_inf - self.q) / 100.0) * self.dt

        return 1 if (self.v_s >= self.v_threshold and v_prev < self.v_threshold) else 0

    def reset(self):
        self.v_s, self.v_d = -60.0, -60.0
        self.h, self.n, self.s, self.c, self.q = 0.9, 0.1, 0.0, 0.0, 0.0
