# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mainen & Sejnowski 1996 — axonal Na spike initiation model

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _safe_exp(x: float) -> float:
    return float(np.exp(np.clip(x, -500.0, 500.0)))


@dataclass
class MainenSejnowskiNeuron:
    """Mainen & Sejnowski 1996 — axonal Na spike initiation model.

    2-compartment: soma (passive) + axon (active Na + K).
    Axon initiates spike via fast Na kinetics; soma follows passively.
    C_s dV_s/dt = -g_L(V_s - E_L) + gc(V_a - V_s) + I
    C_a dV_a/dt = -I_Na - I_K + gc(V_s - V_a)

    Reference: Mainen, Z.F. & Sejnowski, T.J. (1996). Nature 382:363–366.
    """

    vs: float = -65.0
    va: float = -65.0
    m: float = 0.05
    h: float = 0.6
    n: float = 0.3
    kappa: float = 10.0
    g_na: float = 3000.0
    g_k: float = 1500.0
    g_l: float = 1.0
    e_na: float = 50.0
    e_k: float = -90.0
    e_l: float = -70.0
    c_s: float = 1.0
    c_a: float = 0.1
    dt: float = 0.005
    v_threshold: float = -20.0

    def step(self, current: float) -> int:
        vs_prev = self.vs
        for _ in range(20):
            # Axon HH gates (shifted for fast initiation)
            am = 0.182 * (self.va + 25.0) / (1.0 - _safe_exp(-(self.va + 25.0) / 9.0) + 1e-12)
            bm = -0.124 * (self.va + 25.0) / (1.0 - _safe_exp((self.va + 25.0) / 9.0) + 1e-12)
            ah = 0.024 * (self.va + 40.0) / (1.0 - _safe_exp(-(self.va + 40.0) / 5.0) + 1e-12)
            bh = -0.0091 * (self.va + 65.0) / (1.0 - _safe_exp((self.va + 65.0) / 5.0) + 1e-12)
            an = 0.02 * (self.va - 20.0) / (1.0 - _safe_exp(-(self.va - 20.0) / 9.0) + 1e-12)
            bn = -0.002 * (self.va - 20.0) / (1.0 - _safe_exp((self.va - 20.0) / 9.0) + 1e-12)

            self.m = np.clip(self.m + (am * (1 - self.m) - bm * self.m) * self.dt, 0.0, 1.0)
            self.h = np.clip(self.h + (ah * (1 - self.h) - bh * self.h) * self.dt, 0.0, 1.0)
            self.n = np.clip(self.n + (an * (1 - self.n) - bn * self.n) * self.dt, 0.0, 1.0)

            i_na = self.g_na * self.m**3 * self.h * (self.va - self.e_na)
            i_k = self.g_k * self.n * (self.va - self.e_k)
            i_l = self.g_l * (self.vs - self.e_l)

            dvs = (-i_l + self.kappa * (self.va - self.vs) + current) / self.c_s * self.dt
            dva = (-i_na - i_k + self.kappa * (self.vs - self.va)) / self.c_a * self.dt
            self.vs = float(np.clip(self.vs + dvs, -200.0, 200.0))
            self.va = float(np.clip(self.va + dva, -200.0, 200.0))

        return 1 if (self.vs >= self.v_threshold and vs_prev < self.v_threshold) else 0

    def reset(self) -> None:
        self.vs = -65.0
        self.va = -65.0
        self.m, self.h, self.n = 0.05, 0.6, 0.3
