# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Bertram et al. 2008 — phantom burster with dual slow

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class BertramPhantomBurster:
    """Bertram et al. 2008 — phantom burster with dual slow variables.

    C dV/dt  = -(I_Ca + I_K + I_s1 + I_s2 + I_L) + I_ext
    ds1/dt   = (s1_inf(V) - s1) / tau_s1
    ds2/dt   = (s2_inf(V) - s2) / tau_s2

    Two slow variables (s1, s2) with different timescales produce
    bursting via a phantom slow manifold.

    Reference: Bertram, R. et al. (1995). Biophys. J. 68:2323–2332.
    """

    v: float = -50.0
    s1: float = 0.1
    s2: float = 0.1
    g_ca: float = 3.6
    g_k: float = 10.0
    g_s1: float = 4.0
    g_s2: float = 4.0
    g_l: float = 0.2
    e_ca: float = 25.0
    e_k: float = -75.0
    e_l: float = -40.0
    c_m: float = 5.3
    v_m: float = -20.0
    s_m: float = 12.0
    v_n: float = -16.0
    s_n: float = 5.6
    v_s1: float = -40.0
    s_s1: float = 10.0
    v_s2: float = -42.0
    s_s2: float = 0.4
    tau_s1: float = 20000.0
    tau_s2: float = 100000.0
    dt: float = 0.5
    v_threshold: float = -20.0

    def _boltz(self, v: float, vh: float, k: float) -> float:
        return 1.0 / (1.0 + np.exp((vh - v) / k))

    def step(self, current: float) -> int:
        v_prev = self.v
        m_inf = self._boltz(self.v, self.v_m, self.s_m)
        n_inf = self._boltz(self.v, self.v_n, self.s_n)
        s1_inf = self._boltz(self.v, self.v_s1, self.s_s1)
        s2_inf = self._boltz(self.v, self.v_s2, self.s_s2)

        i_ca = self.g_ca * m_inf * (self.v - self.e_ca)
        i_k = self.g_k * n_inf * (self.v - self.e_k)
        i_s1 = self.g_s1 * self.s1 * (self.v - self.e_k)
        i_s2 = self.g_s2 * self.s2 * (self.v - self.e_k)
        i_l = self.g_l * (self.v - self.e_l)

        self.v += (-i_ca - i_k - i_s1 - i_s2 - i_l + current) / self.c_m * self.dt
        self.s1 += (s1_inf - self.s1) / self.tau_s1 * self.dt
        self.s2 += (s2_inf - self.s2) / self.tau_s2 * self.dt

        return 1 if (self.v >= self.v_threshold and v_prev < self.v_threshold) else 0

    def reset(self) -> None:
        self.v = -50.0
        self.s1 = 0.1
        self.s2 = 0.1
