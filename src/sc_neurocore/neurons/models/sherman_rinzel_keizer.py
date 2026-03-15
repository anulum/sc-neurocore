# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class ShermanRinzelKeizerNeuron:
    """Sherman, Rinzel & Keizer 1988 — pancreatic beta cell (reduced)."""

    v: float = -50.0
    n: float = 0.1
    s: float = 0.1
    g_ca: float = 3.6
    g_k: float = 10.0
    g_s: float = 4.0
    e_ca: float = 25.0
    e_k: float = -75.0
    tau_s: float = 5000.0
    dt: float = 0.5
    v_threshold: float = -20.0

    def step(self, current: float) -> int:
        v_prev = self.v
        m_inf = 1.0 / (1.0 + np.exp(-(self.v + 20.0) / 12.0))
        n_inf = 1.0 / (1.0 + np.exp(-(self.v + 16.0) / 5.0))
        s_inf = 1.0 / (1.0 + np.exp(-(self.v + 35.0) / 10.0))
        tau_n = 9.09
        i_ca = self.g_ca * m_inf * (self.v - self.e_ca)
        i_k = self.g_k * self.n * (self.v - self.e_k)
        i_s = self.g_s * self.s * (self.v - self.e_k)
        self.v += (-i_ca - i_k - i_s + current) * self.dt
        self.n += (n_inf - self.n) / tau_n * self.dt
        self.s += (s_inf - self.s) / self.tau_s * self.dt
        return 1 if (self.v >= self.v_threshold and v_prev < self.v_threshold) else 0

    def reset(self):
        self.v, self.n, self.s = -50.0, 0.1, 0.1


# ── IF VARIANTS ────────────────────────────────────────────────────
