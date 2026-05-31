# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Terman & Wang 1995 — relaxation oscillator for LEGION

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class TermanWangOscillator:
    """Terman & Wang 1995 — relaxation oscillator for LEGION networks.

    dv/dt = f(v) - w + I + rho
    dw/dt = epsilon * (g(v) - w)

    f(v) = 3*v - v^3 + 2                (cubic nullcline)
    g(v) = alpha * (1 + tanh(v/beta))    (sigmoid recovery)

    Reference: Terman, D. & Wang, D.L. (1995). Neural Comput. 7:507–517.
    """

    v: float = -1.5
    w: float = -0.5
    alpha: float = 3.0
    beta: float = 0.2
    epsilon: float = 0.02
    rho: float = 0.0
    dt: float = 0.05
    v_peak: float = 1.5

    def __post_init__(self) -> None:
        for name in (
            "v",
            "w",
            "alpha",
            "beta",
            "epsilon",
            "rho",
            "dt",
            "v_peak",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            setattr(self, name, value)
        for name in ("beta", "epsilon", "dt"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")

    @staticmethod
    def _validate_state(v: float, w: float) -> tuple[float, float]:
        v_value = float(v)
        w_value = float(w)
        if not math.isfinite(v_value) or not math.isfinite(w_value):
            raise FloatingPointError("Terman-Wang runtime state must be finite")
        return v_value, w_value

    def step(self, current: float) -> int:
        drive = float(current)
        if not math.isfinite(drive):
            raise ValueError("current must be finite")

        v, w = self._validate_state(self.v, self.w)
        v_prev = v
        try:
            f = 3.0 * v - v**3 + 2.0
        except OverflowError as exc:
            raise FloatingPointError("Terman-Wang cubic nullcline overflowed") from exc
        g = self.alpha * (1.0 + math.tanh(v / self.beta))
        dv = (f - w + drive + self.rho) * self.dt
        dw = self.epsilon * (g - w) * self.dt
        if not math.isfinite(dv) or not math.isfinite(dw):
            raise FloatingPointError("Terman-Wang update became non-finite")

        next_v = v + dv
        next_w = w + dw
        self.v, self.w = self._validate_state(next_v, next_w)
        return 1 if (self.v >= self.v_peak and v_prev < self.v_peak) else 0

    def reset(self) -> None:
        self.v = -1.5
        self.w = -0.5
