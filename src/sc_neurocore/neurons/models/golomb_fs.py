# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Golomb et al. 2007 — fast-spiking interneuron with Kv3

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GolombFSNeuron:
    """Golomb et al. 2007 — fast-spiking interneuron with Kv3.

    C dV/dt = -I_Na - I_Kd - I_Kv3 - I_L + I_ext
    Kv3 channel enables narrow spikes and high sustained firing.

    Reference: Golomb, D. et al. (2007). J. Neurophysiol. 97:3831–3843.
    """

    v: float = -65.0
    h: float = 0.9
    n: float = 0.1
    p: float = 0.0
    g_na: float = 112.5
    g_kd: float = 225.0
    g_kv3: float = 150.0
    g_l: float = 0.25
    e_na: float = 50.0
    e_k: float = -90.0
    e_l: float = -70.0
    c_m: float = 1.0
    dt: float = 0.01
    v_threshold: float = -20.0

    def step(self, current: float) -> int:
        v_prev = self.v
        for _ in range(10):
            m_inf = 1.0 / (1.0 + np.exp(-(self.v + 24.0) / 11.5))
            h_inf = 1.0 / (1.0 + np.exp((self.v + 58.3) / 6.7))
            tau_h = 0.5 + 14.0 / (1.0 + np.exp((self.v + 60.0) / 12.0))
            n_inf = 1.0 / (1.0 + np.exp(-(self.v + 12.4) / 6.8))
            tau_n = 0.087 + 11.4 / (1.0 + np.exp((self.v + 14.6) / 8.6))
            # Kv3: fast activating, high threshold
            p_inf = 1.0 / (1.0 + np.exp(-(self.v + 3.0) / 8.0))
            tau_p = 0.1 + 4.0 / (1.0 + np.exp((self.v + 25.0) / 10.0))

            self.h += (h_inf - self.h) / tau_h * self.dt
            self.n += (n_inf - self.n) / tau_n * self.dt
            self.p += (p_inf - self.p) / tau_p * self.dt

            i_na = self.g_na * m_inf**3 * self.h * (self.v - self.e_na)
            i_kd = self.g_kd * self.n**4 * (self.v - self.e_k)
            i_kv3 = self.g_kv3 * self.p**2 * (self.v - self.e_k)
            i_l = self.g_l * (self.v - self.e_l)

            self.v += (-i_na - i_kd - i_kv3 - i_l + current) / self.c_m * self.dt

        return 1 if (self.v >= self.v_threshold and v_prev < self.v_threshold) else 0

    def reset(self) -> None:
        self.v = -65.0
        self.h, self.n, self.p = 0.9, 0.1, 0.0
