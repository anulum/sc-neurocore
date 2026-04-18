# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Breakspear, Terry & Friston 2003 — neural mass with ion

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class LarterBreakspearNeuron:
    """Breakspear, Terry & Friston 2003 — neural mass with ion channels.

    3 ODEs per node. Combines Wilson-Cowan population dynamics with
    conductance-based ion channel kinetics for whole-brain modelling.
    Used in The Virtual Brain (TVB) simulator.

    Reference: Larter, R. et al. (1999). Chaos 9:795–804.; Breakspear, M. et al. (2003). Cereb. Cortex 13:189–202.
    """

    v: float = -0.5
    w: float = 0.0
    z: float = 0.0
    g_ca: float = 1.1
    g_na: float = 6.7
    g_k: float = 2.0
    v_ca: float = 1.0
    v_na: float = 0.53
    v_k: float = -0.7
    v_l: float = -0.5
    g_l: float = 0.5
    phi: float = 0.7
    tau_k: float = 1.0
    b: float = 0.1
    a_ee: float = 0.36
    v0: float = 0.0
    i_ext: float = 0.3
    dt: float = 0.01

    def _m_ca(self, v: float) -> Any:
        return 0.5 * (1.0 + np.tanh((v - (-0.01)) / 0.15))

    def _m_na(self, v: float) -> Any:
        return 0.5 * (1.0 + np.tanh((v - 0.12) / 0.15))

    def _m_k(self, v: float) -> Any:
        return 0.5 * (1.0 + np.tanh((v - self.v0) / 0.3))

    def step(self, coupling: float = 0.0) -> float:
        i_ca = self.g_ca * self._m_ca(self.v) * (self.v - self.v_ca)
        i_na = self.g_na * self._m_na(self.v) * (self.v - self.v_na)
        i_k = self.g_k * self.w * (self.v - self.v_k)
        i_l = self.g_l * (self.v - self.v_l)

        dv = -i_ca - i_na - i_k - i_l + self.i_ext + coupling + self.a_ee * self.v
        dw = self.phi * (self._m_k(self.v) - self.w) / self.tau_k
        dz = self.b * (self.v + 0.5 - self.z)

        self.v += dv * self.dt
        self.w += dw * self.dt
        self.z += dz * self.dt

        return self.v

    def reset(self) -> None:
        self.v, self.w, self.z = -0.5, 0.0, 0.0
