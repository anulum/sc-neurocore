# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Chay 1985 — pancreatic beta cell burster

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class ChayNeuron:
    """Chay 1985 — pancreatic beta cell burster.

    Reference: Chay, T.R. (1985). Physica D 16:233–242.
    """

    v: float = -50.0
    n: float = 0.1
    ca: float = 0.1
    g_ca: float = 25.0
    g_k: float = 1400.0
    g_kca: float = 12.0
    g_l: float = 7.0
    e_ca: float = 100.0
    e_k: float = -75.0
    e_l: float = -40.0
    rho: float = 0.00015
    alpha_ca: float = 0.002
    k_ca: float = 0.04
    dt: float = 0.02
    v_threshold: float = -20.0

    def step(self, current: float) -> int:
        v_prev = self.v
        m_inf = 1.0 / (1.0 + np.exp(np.clip(-(self.v + 25.0) / 8.0, -500.0, 500.0)))
        n_inf = 1.0 / (1.0 + np.exp(np.clip(-(self.v + 18.0) / 14.0, -500.0, 500.0)))
        tau_n = 1.0 / (0.01 * max(abs(self.v + 18.0), 0.01))
        i_ca = self.g_ca * m_inf * (self.v - self.e_ca)
        kca_act = self.ca / (self.ca + 1.0)
        i_k = self.g_k * self.n * (self.v - self.e_k)
        i_kca = self.g_kca * kca_act * (self.v - self.e_k)
        i_l = self.g_l * (self.v - self.e_l)
        self.v += (-i_ca - i_k - i_kca - i_l + current) * self.dt
        self.v = np.clip(self.v, -200.0, 200.0)
        self.n += (n_inf - self.n) / max(tau_n, 0.01) * self.dt
        self.n = np.clip(self.n, 0.0, 1.0)
        self.ca = max(
            0.0, self.ca + self.rho * (-self.alpha_ca * i_ca - self.k_ca * self.ca) * self.dt
        )
        return 1 if (self.v >= self.v_threshold and v_prev < self.v_threshold) else 0

    def reset(self) -> None:
        self.v, self.n, self.ca = -50.0, 0.1, 0.1
