# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Ibarz et al. 2007 / Tanaka — piecewise-linear bursting map

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class IbarzTanakaMapNeuron:
    """Ibarz et al. 2007 / Tanaka — piecewise-linear bursting map.

    x(n+1) = f(x(n)) + y(n) + I
    y(n+1) = y(n) - mu*(x(n) + 1) + mu*sigma

    f(x) = alpha/(1-x)       if x <= 0
         = alpha + beta*x     if 0 < x < alpha+beta (spiking)
    Reset x -> x_reset when x >= x_threshold.
    """

    x: float = -1.0
    y: float = -2.5
    alpha: float = 3.65
    beta: float = 0.25
    mu: float = 0.0005
    sigma: float = -1.6
    x_threshold: float = 3.0
    x_reset: float = -1.0

    def _f(self, x: float) -> float:
        if x <= 0.0:
            return self.alpha / (1.0 - x)
        return self.alpha + self.beta * x

    def step(self, current: float) -> int:
        x_new = self._f(self.x) + self.y + current
        y_new = self.y - self.mu * (self.x + 1.0) + self.mu * self.sigma
        self.x = x_new
        self.y = y_new
        if self.x >= self.x_threshold:
            self.x = self.x_reset
            return 1
        return 0

    def reset(self) -> None:
        self.x = -1.0
        self.y = -2.5
