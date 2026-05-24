# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Resonate-and-fire neuron model

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class ResonateAndFireNeuron:
    x: float = 0.0
    y: float = 0.0
    b: float = -0.1
    omega: float = 1.0
    threshold: float = 1.0
    dt: float = 0.05

    def __post_init__(self) -> None:
        for name in ("x", "y", "b"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if not math.isfinite(self.omega) or self.omega <= 0.0:
            raise ValueError("omega must be finite and positive")
        for name in ("threshold", "dt"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")

    def step(self, current: float) -> int:
        if not math.isfinite(current):
            raise ValueError("current must be finite")

        dx = (self.b * self.x - self.omega * self.y + current) * self.dt
        dy = (self.omega * self.x + self.b * self.y) * self.dt
        next_x = self.x + dx
        next_y = self.y + dy
        radius = math.hypot(next_x, next_y)
        if not all(math.isfinite(value) for value in (dx, dy, next_x, next_y, radius)):
            raise ValueError("Euler update must be finite")

        self.x = next_x
        self.y = next_y
        if radius >= self.threshold:
            self.x = 0.0
            self.y = 0.0
            return 1
        return 0

    def reset(self) -> None:
        self.x = 0.0
        self.y = 0.0
