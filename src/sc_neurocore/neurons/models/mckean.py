# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class McKeanNeuron:
    """McKean 1970 — piecewise-linear FitzHugh-Nagumo caricature.

    dv/dt = f(v) - w + I
    dw/dt = epsilon * (v - gamma*w)

    f(v) = -v          if v < a/2
         = v - a       if a/2 <= v < (1+a)/2
         = 1 - v       if v >= (1+a)/2
    """

    v: float = 0.0
    w: float = 0.0
    a: float = 0.25
    epsilon: float = 0.01
    gamma: float = 0.5
    dt: float = 0.1
    v_peak: float = 0.8

    def _f(self, v: float) -> float:
        mid1 = self.a / 2.0
        mid2 = (1.0 + self.a) / 2.0
        if v < mid1:
            return -v
        elif v < mid2:
            return v - self.a
        else:
            return 1.0 - v

    def step(self, current: float) -> int:
        dv = (self._f(self.v) - self.w + current) * self.dt
        dw = self.epsilon * (self.v - self.gamma * self.w) * self.dt
        v_prev = self.v
        self.v += dv
        self.w += dw
        return 1 if (self.v >= self.v_peak and v_prev < self.v_peak) else 0

    def reset(self):
        self.v = 0.0
        self.w = 0.0
