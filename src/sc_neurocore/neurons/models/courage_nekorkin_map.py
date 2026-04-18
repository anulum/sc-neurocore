# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Courbage, Nekorkin & Vdovin 2007 — piecewise-linear

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CourageNekorkinMapNeuron:
    """Courbage, Nekorkin & Vdovin 2007 — piecewise-linear Lorenz-type map."""

    x: float = 0.0
    y: float = 0.0
    alpha: float = 3.0
    beta: float = 0.001
    j: float = 0.1
    x_threshold: float = 1.0

    def _f(self, x: float) -> float:
        if x < 0:
            return self.alpha * x
        return self.alpha * x / (1.0 + self.alpha * x)

    def step(self, current: float = 0.0) -> int:
        x_prev = self.x
        x_new = self._f(self.x) + self.y + current + self.j
        y_new = self.y - self.beta * (self.x + 1.0)
        # Clip to prevent divergence (map can escape without bounds)
        self.x = max(min(x_new, 1e6), -1e6)
        self.y = max(min(y_new, 1e6), -1e6)
        return 1 if (self.x >= self.x_threshold and x_prev < self.x_threshold) else 0

    def reset(self) -> None:
        self.x, self.y = 0.0, 0.0
