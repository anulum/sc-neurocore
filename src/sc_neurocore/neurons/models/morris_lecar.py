# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Morris-Lecar 1981 — calcium-potassium oscillator

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import math


@dataclass
class MorrisLecarNeuron:
    """Morris-Lecar 1981 — calcium-potassium oscillator.

    C dv/dt = -g_Ca m_∞(v)(v-E_Ca) - g_K w(v-E_K) - g_L(v-E_L) + I
    dw/dt = λ(v)(w_∞(v) - w)

    Reference: Morris, C. & Lecar, H. (1981). Biophys. J. 35:193–213.
    """

    v: float = -60.0
    w: float = 0.0
    c_m: float = 20.0
    g_ca: float = 4.0
    g_k: float = 8.0
    g_l: float = 2.0
    e_ca: float = 120.0
    e_k: float = -84.0
    e_l: float = -60.0
    v1: float = -1.2
    v2: float = 18.0
    v3: float = 12.0
    v4: float = 17.4
    phi: float = 1.0 / 15.0
    dt: float = 0.1
    v_threshold: float = 0.0

    def _m_inf(self, v: float) -> Any:
        return 0.5 * (1.0 + math.tanh((v - self.v1) / self.v2))

    def _w_inf(self, v: float) -> Any:
        return 0.5 * (1.0 + math.tanh((v - self.v3) / self.v4))

    def _lam(self, v: float) -> Any:
        return self.phi * math.cosh((v - self.v3) / (2.0 * self.v4))

    def step(self, current: float) -> int:
        v_prev = self.v
        m_inf = self._m_inf(self.v)
        w_inf = self._w_inf(self.v)
        lam = self._lam(self.v)

        i_ca = self.g_ca * m_inf * (self.v - self.e_ca)
        i_k = self.g_k * self.w * (self.v - self.e_k)
        i_l = self.g_l * (self.v - self.e_l)

        self.v += (-i_ca - i_k - i_l + current) / self.c_m * self.dt
        self.w += lam * (w_inf - self.w) * self.dt

        return 1 if (self.v >= self.v_threshold and v_prev < self.v_threshold) else 0

    def reset(self) -> None:
        self.v = -60.0
        self.w = 0.0
