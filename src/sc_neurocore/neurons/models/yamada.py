# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Yamada, Kashimori & Kambara 1989 — subcritical Hopf burster

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class YamadaNeuron:
    """Yamada, Kashimori & Kambara 1989 — subcritical Hopf burster.

    3 ODEs: V, n (fast K recovery), q (slow variable for bursting).
    Exhibits square-wave bursting via slow modulation of a Hopf bifurcation.
    """

    v: float = -60.0
    n: float = 0.1
    q: float = 0.0  # slow variable
    g_na: float = 20.0
    g_k: float = 10.0
    g_q: float = 5.0  # slow current conductance
    g_l: float = 0.5
    e_na: float = 60.0
    e_k: float = -80.0
    e_q: float = -80.0
    e_l: float = -60.0
    tau_q: float = 300.0  # ms, slow timescale
    dt: float = 0.05
    v_threshold: float = -20.0

    def step(self, current: float) -> int:
        v_prev = self.v
        m_inf = 1.0 / (1.0 + np.exp(-(self.v + 30.0) / 9.5))
        n_inf = 1.0 / (1.0 + np.exp(-(self.v + 30.0) / 10.0))
        q_inf = 1.0 / (1.0 + np.exp(-(self.v + 50.0) / 10.0))
        tau_n = 1.0 + 7.5 / (1.0 + np.exp((self.v + 40.0) / 12.0))

        i_na = self.g_na * m_inf**3 * (1.0 - self.n) * (self.v - self.e_na)
        i_k = self.g_k * self.n**4 * (self.v - self.e_k)
        i_q = self.g_q * self.q * (self.v - self.e_q)
        i_l = self.g_l * (self.v - self.e_l)

        self.v += (-i_na - i_k - i_q - i_l + current) * self.dt
        self.n += (n_inf - self.n) / tau_n * self.dt
        self.q += (q_inf - self.q) / self.tau_q * self.dt

        return 1 if (self.v >= self.v_threshold and v_prev < self.v_threshold) else 0

    def reset(self) -> None:
        self.v, self.n, self.q = -60.0, 0.1, 0.0
