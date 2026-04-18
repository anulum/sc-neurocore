# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Hodgkin-Huxley 1952 — 4-ODE ion channel model

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class HodgkinHuxleyNeuron:
    """Hodgkin-Huxley 1952 — 4-ODE ion channel model.

    C_m dv/dt = -g_Na m³h(v-E_Na) - g_K n⁴(v-E_K) - g_L(v-E_L) + I
    dm/dt = α_m(1-m) - β_m·m
    dh/dt = α_h(1-h) - β_h·h
    dn/dt = α_n(1-n) - β_n·n
    """

    v: float = -65.0
    m: float = 0.05
    h: float = 0.6
    n: float = 0.32
    c_m: float = 1.0
    g_na: float = 120.0
    g_k: float = 36.0
    g_l: float = 0.3
    e_na: float = 50.0
    e_k: float = -77.0
    e_l: float = -54.4
    dt: float = 0.01
    v_threshold: float = 0.0

    def _alpha_m(self, v: float) -> float:
        d = v + 40.0
        if abs(d) < 1e-7:
            return 1.0
        return 0.1 * d / (1.0 - np.exp(-d / 10.0))

    def _beta_m(self, v: float) -> float:
        return float(4.0 * np.exp(-(v + 65.0) / 18.0))

    def _alpha_h(self, v: float) -> float:
        return float(0.07 * np.exp(-(v + 65.0) / 20.0))

    def _beta_h(self, v: float) -> float:
        return float(1.0 / (1.0 + np.exp(-(v + 35.0) / 10.0)))

    def _alpha_n(self, v: float) -> float:
        d = v + 55.0
        if abs(d) < 1e-7:
            return 0.1
        return 0.01 * d / (1.0 - np.exp(-d / 10.0))

    def _beta_n(self, v: float) -> float:
        return float(0.125 * np.exp(-(v + 65.0) / 80.0))

    def step(self, current: float) -> int:
        v_prev = self.v
        for _ in range(round(1.0 / self.dt)):
            am, bm = self._alpha_m(self.v), self._beta_m(self.v)
            ah, bh = self._alpha_h(self.v), self._beta_h(self.v)
            an, bn = self._alpha_n(self.v), self._beta_n(self.v)

            self.m += (am * (1 - self.m) - bm * self.m) * self.dt
            self.h += (ah * (1 - self.h) - bh * self.h) * self.dt
            self.n += (an * (1 - self.n) - bn * self.n) * self.dt

            i_na = self.g_na * self.m**3 * self.h * (self.v - self.e_na)
            i_k = self.g_k * self.n**4 * (self.v - self.e_k)
            i_l = self.g_l * (self.v - self.e_l)

            self.v += (-i_na - i_k - i_l + current) / self.c_m * self.dt

        return 1 if (self.v >= self.v_threshold and v_prev < self.v_threshold) else 0

    def reset(self) -> None:
        self.v = -65.0
        self.m = 0.05
        self.h = 0.6
        self.n = 0.32
