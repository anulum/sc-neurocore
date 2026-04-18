# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Wang-Buzsáki 1996 — fast-spiking GABAergic interneuron

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class WangBuzsakiNeuron:
    """Wang-Buzsáki 1996 — fast-spiking GABAergic interneuron.

    3 ODEs. Simplified HH with only Na + K delayed rectifier.
    Designed for gamma (30-80 Hz) oscillation modelling.
    """

    v: float = -65.0
    h: float = 0.8
    n: float = 0.1
    g_na: float = 35.0
    g_k: float = 9.0
    g_l: float = 0.1
    e_na: float = 55.0
    e_k: float = -90.0
    e_l: float = -65.0
    c_m: float = 1.0
    phi: float = 5.0
    dt: float = 0.01
    v_threshold: float = -20.0

    def step(self, current: float) -> int:
        v_prev = self.v
        for _ in range(int(0.5 / max(self.dt, 0.001))):
            # m is instantaneous (m_inf)
            alpha_m = (
                0.1 * (self.v + 35.0) / (1.0 - np.exp(-(self.v + 35.0) / 10.0))
                if abs(self.v + 35.0) > 1e-6
                else 1.0
            )
            beta_m = 4.0 * np.exp(-(self.v + 60.0) / 18.0)
            m_inf = alpha_m / (alpha_m + beta_m)

            alpha_h = 0.07 * np.exp(-(self.v + 58.0) / 20.0)
            beta_h = 1.0 / (1.0 + np.exp(-(self.v + 28.0) / 10.0))
            alpha_n = (
                0.01 * (self.v + 34.0) / (1.0 - np.exp(-(self.v + 34.0) / 10.0))
                if abs(self.v + 34.0) > 1e-6
                else 0.1
            )
            beta_n = 0.125 * np.exp(-(self.v + 44.0) / 80.0)

            self.h += self.phi * (alpha_h * (1 - self.h) - beta_h * self.h) * self.dt
            self.n += self.phi * (alpha_n * (1 - self.n) - beta_n * self.n) * self.dt

            i_na = self.g_na * m_inf**3 * self.h * (self.v - self.e_na)
            i_k = self.g_k * self.n**4 * (self.v - self.e_k)
            i_l = self.g_l * (self.v - self.e_l)

            self.v += (-i_na - i_k - i_l + current) / self.c_m * self.dt

        return 1 if (self.v >= self.v_threshold and v_prev < self.v_threshold) else 0

    def reset(self) -> None:
        self.v = -65.0
        self.h, self.n = 0.8, 0.1
