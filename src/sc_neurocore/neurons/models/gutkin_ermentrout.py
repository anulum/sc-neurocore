# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Gutkin & Ermentrout 1998 — persistent Na + K minimal cond...

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class GutkinErmentroutNeuron:
    """Gutkin & Ermentrout 1998 — persistent Na + K minimal conductance."""

    v: float = -65.0
    n: float = 0.1
    g_na: float = 20.0
    g_k: float = 10.0
    g_l: float = 8.0
    e_na: float = 60.0
    e_k: float = -90.0
    e_l: float = -80.0
    dt: float = 0.05
    v_threshold: float = -20.0

    def step(self, current: float) -> int:
        v_prev = self.v
        m_inf = 1.0 / (1.0 + np.exp(-(self.v + 20.0) / 15.0))
        n_inf = 1.0 / (1.0 + np.exp(-(self.v + 25.0) / 5.0))
        tau_n = 1.0
        self.n += (n_inf - self.n) / tau_n * self.dt
        i_na = self.g_na * m_inf * (self.v - self.e_na)
        i_k = self.g_k * self.n * (self.v - self.e_k)
        i_l = self.g_l * (self.v - self.e_l)
        self.v += (-i_na - i_k - i_l + current) * self.dt
        return 1 if (self.v >= self.v_threshold and v_prev < self.v_threshold) else 0

    def reset(self):
        self.v = -65.0
        self.n = 0.1


# ── BURSTING MODELS ────────────────────────────────────────────────
