# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Rulkov 2001 — discrete map-based neuron (no ODE, O(1)

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class RulkovMapNeuron:
    """Rulkov 2001 — discrete map-based neuron (no ODE, O(1) per step).

    x[n+1] = f(x[n], y[n]) + I
    y[n+1] = y[n] - μ(x[n] + 1) + μσ
    Fast iteration, exhibits spiking and bursting.

    Reference: Rulkov, N.F. (2002). Phys. Rev. E 65:041922.
    """

    x: float = -1.0
    y: float = -3.0
    alpha: float = 4.0
    sigma: float = -1.6
    mu: float = 0.001
    x_threshold: float = 0.0

    def __post_init__(self) -> None:
        for name in ("x", "y", "alpha", "sigma", "mu", "x_threshold"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            setattr(self, name, value)
        if self.alpha <= 0.0:
            raise ValueError("alpha must be positive")
        if self.mu <= 0.0:
            raise ValueError("mu must be positive")

    @staticmethod
    def _validate_state(x: float, y: float) -> tuple[float, float]:
        x_value = float(x)
        y_value = float(y)
        if not math.isfinite(x_value) or not math.isfinite(y_value):
            raise FloatingPointError("Rulkov map state must be finite")
        return x_value, y_value

    def step(self, current: float = 0.0) -> int:
        drive = float(current)
        if not math.isfinite(drive):
            raise ValueError("current must be finite")

        x, y = self._validate_state(self.x, self.y)
        x_prev = x
        branch_boundary = self.alpha + y + drive
        if not math.isfinite(branch_boundary):
            raise FloatingPointError("Rulkov map branch boundary became non-finite")
        if x <= 0:
            denominator = 1.0 - x
            if denominator <= 0.0 or not math.isfinite(denominator):
                raise FloatingPointError("Rulkov map branch denominator is invalid")
            x_new = self.alpha / denominator + y + drive
        elif x < branch_boundary:
            x_new = branch_boundary
        else:
            x_new = -1.0
        y_new = y - self.mu * (x + 1.0) + self.mu * self.sigma
        if not math.isfinite(x_new) or not math.isfinite(y_new):
            raise FloatingPointError("Rulkov map candidate state became non-finite")
        self.x = x_new
        self.y = y_new
        return 1 if (self.x >= self.x_threshold and x_prev < self.x_threshold) else 0

    def reset(self) -> None:
        self.x, self.y = -1.0, -3.0
