# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — McKean 1970 — piecewise-linear FitzHugh-Nagumo caricature

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class McKeanNeuron:
    """McKean 1970 — piecewise-linear FitzHugh-Nagumo caricature.

    dv/dt = f(v) - w + I
    dw/dt = epsilon * (v - gamma*w)

    f(v) = -v          if v < a/2
         = v - a       if a/2 <= v < (1+a)/2
         = 1 - v       if v >= (1+a)/2

    Reference: McKean, H.P. (1970). Adv. Math. 4:209–223.
    """

    v: float = 0.0
    w: float = 0.0
    a: float = 0.25
    epsilon: float = 0.01
    gamma: float = 0.5
    dt: float = 0.1
    v_peak: float = 0.8

    def __post_init__(self) -> None:
        for name in ("v", "w", "v_peak"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if not math.isfinite(self.a) or not 0.0 < self.a < 1.0:
            raise ValueError("a must be finite and in the open interval (0, 1)")
        for name in ("epsilon", "gamma", "dt"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")

    def _f(self, v: float) -> float:
        if not math.isfinite(v):
            raise FloatingPointError("McKean voltage became non-finite")
        mid1 = self.a / 2.0
        mid2 = (1.0 + self.a) / 2.0
        if v < mid1:
            return -v
        elif v < mid2:
            return v - self.a
        else:
            return 1.0 - v

    @staticmethod
    def _validate_state(v: float, w: float) -> tuple[float, float]:
        if not (math.isfinite(v) and math.isfinite(w)):
            raise FloatingPointError("McKean state became non-finite")
        return float(v), float(w)

    def step(self, current: float) -> int:
        if not math.isfinite(current):
            raise ValueError("current must be finite")

        dv = (self._f(self.v) - self.w + current) * self.dt
        dw = self.epsilon * (self.v - self.gamma * self.w) * self.dt
        v_prev = self.v
        self.v, self.w = self._validate_state(self.v + dv, self.w + dw)
        return 1 if (self.v >= self.v_peak and v_prev < self.v_peak) else 0

    def reset(self) -> None:
        self.v = 0.0
        self.w = 0.0
