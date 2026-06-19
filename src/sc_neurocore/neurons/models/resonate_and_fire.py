# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Resonate-and-fire neuron model

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class ResonateAndFireNeuron:
    x: float = 0.0
    y: float = 0.0
    b: float = -0.1
    omega: float = 1.0
    threshold: float = 1.0
    dt: float = 0.05

    def __post_init__(self) -> None:
        self._validate_runtime_state()

    def step(self, current: float) -> int:
        if not math.isfinite(current):
            raise ValueError("current must be finite")
        self._validate_runtime_state()

        next_x, next_y = self._exact_linear_flow(
            self.x,
            self.y,
            current,
            self.b,
            self.omega,
            self.dt,
        )
        radius = math.hypot(next_x, next_y)
        if not all(math.isfinite(value) for value in (next_x, next_y, radius)):
            raise ValueError("exact resonator update must be finite")

        if radius >= self.threshold:
            self.x = 0.0
            self.y = 0.0
            return 1

        self.x = next_x
        self.y = next_y
        return 0

    def reset(self) -> None:
        self.x = 0.0
        self.y = 0.0

    def _validate_runtime_state(self) -> None:
        for name in ("x", "y", "b"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if not math.isfinite(self.omega) or self.omega <= 0.0:
            raise ValueError("omega must be finite and positive")
        for name in ("threshold", "dt"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")

    @staticmethod
    def _exact_linear_flow(
        x: float,
        y: float,
        current: float,
        b: float,
        omega: float,
        dt: float,
    ) -> tuple[float, float]:
        denominator = b * b + omega * omega
        damping_argument = b * dt
        angle = omega * dt
        if not all(math.isfinite(value) for value in (denominator, damping_argument, angle)):
            raise ValueError("exact resonator coefficients must be finite")
        if denominator <= 0.0:
            raise ValueError("exact resonator denominator must be positive")

        x_ss = -b * current / denominator
        y_ss = omega * current / denominator
        if not math.isfinite(x_ss) or not math.isfinite(y_ss):
            raise ValueError("exact resonator equilibrium must be finite")

        try:
            decay = math.exp(damping_argument)
        except OverflowError as exc:
            raise ValueError("exact resonator decay must be finite") from exc
        cos_angle = math.cos(angle)
        sin_angle = math.sin(angle)

        dx = x - x_ss
        dy = y - y_ss
        next_x = x_ss + decay * (dx * cos_angle - dy * sin_angle)
        next_y = y_ss + decay * (dx * sin_angle + dy * cos_angle)
        return next_x, next_y
