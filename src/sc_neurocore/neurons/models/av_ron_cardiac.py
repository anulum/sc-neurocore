# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Av-Ron, Parnas & Segel 1993 — cardiac ganglion Type III b...

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class AvRonCardiacNeuron:
    """Av-Ron, Parnas & Segel 1993 — cardiac ganglion Type III bursting.

    Reduced HH-type with slow inactivation producing plateau bursts.
    """

    v: float = -60.0
    h: float = 0.6
    n: float = 0.3
    s: float = 0.5  # slow inactivation
    g_na: float = 80.0
    g_k: float = 40.0
    g_s: float = 20.0  # slow current
    g_l: float = 0.1
    e_na: float = 40.0
    e_k: float = -80.0
    e_s: float = -25.0
    e_l: float = -60.0
    dt: float = 0.02
    v_threshold: float = -20.0

    def step(self, current: float) -> int:
        v_prev = self.v
        m_inf = 1.0 / (1.0 + np.exp(-(self.v + 40.0) / 7.0))
        h_inf = 1.0 / (1.0 + np.exp((self.v + 45.0) / 5.0))
        n_inf = 1.0 / (1.0 + np.exp(-(self.v + 40.0) / 15.0))
        s_inf = 1.0 / (1.0 + np.exp((self.v + 35.0) / 3.0))

        tau_h = 1.0 + 12.0 / (1.0 + np.exp((self.v + 50.0) / 8.0))
        tau_n = 1.0 + 8.0 / (1.0 + np.exp((self.v + 35.0) / 8.0))
        tau_s = 200.0 + 1000.0 / (1.0 + np.exp((self.v + 30.0) / 5.0))

        self.h += (h_inf - self.h) / tau_h * self.dt
        self.n += (n_inf - self.n) / tau_n * self.dt
        self.s += (s_inf - self.s) / tau_s * self.dt

        i_na = self.g_na * m_inf**3 * self.h * (self.v - self.e_na)
        i_k = self.g_k * self.n**4 * (self.v - self.e_k)
        i_s = self.g_s * self.s * (self.v - self.e_s)
        i_l = self.g_l * (self.v - self.e_l)

        self.v += (-i_na - i_k - i_s - i_l + current) * self.dt
        return 1 if (self.v >= self.v_threshold and v_prev < self.v_threshold) else 0

    def reset(self):
        self.v, self.h, self.n, self.s = -60.0, 0.6, 0.3, 0.5
