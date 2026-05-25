# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Hindmarsh-Rose 1984 — 3D chaotic bursting model

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import math


@dataclass
class HindmarshRoseNeuron:
    """Hindmarsh-Rose 1984 — 3D chaotic bursting model.

    dx/dt = y - x³ + bx² - z + I
    dy/dt = 1 - 5x² - y
    dz/dt = r(s(x - x_rest) - z)

    Reference: Hindmarsh, J.L. & Rose, R.M. (1984). Proc. R. Soc. Lond. B 221:87–102.
    """

    x: float = -1.6
    y: float = -10.0
    z: float = 2.0
    b: float = 3.0
    r: float = 0.001
    s: float = 4.0
    x_rest: float = -1.6
    dt: float = 0.1
    x_threshold: float = 1.0
    integrator: Literal["rk4", "euler"] = "rk4"

    def __post_init__(self) -> None:
        if self.integrator not in {"rk4", "euler"}:
            raise ValueError("integrator must be 'rk4' or 'euler'")
        for name in ("x", "y", "z", "b", "r", "s", "x_rest", "dt", "x_threshold"):
            value = getattr(self, name)
            if not isinstance(value, int | float) or not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
            setattr(self, name, float(value))
        for name in ("dt", "r", "s"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")

    def _derivatives(self, x: float, y: float, z: float, current: float) -> tuple[float, float, float]:
        dx = y - x**3 + self.b * x**2 - z + current
        dy = 1.0 - 5.0 * x**2 - y
        dz = self.r * (self.s * (x - self.x_rest) - z)
        return dx, dy, dz

    def _set_state(self, x: float, y: float, z: float) -> None:
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
            raise FloatingPointError("Hindmarsh-Rose state became non-finite")
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def _step_euler(self, current: float) -> None:
        dx, dy, dz = self._derivatives(self.x, self.y, self.z, current)
        self._set_state(self.x + dx * self.dt, self.y + dy * self.dt, self.z + dz * self.dt)

    def _step_rk4(self, current: float) -> None:
        x0, y0, z0 = self.x, self.y, self.z
        dt = self.dt
        k1 = self._derivatives(x0, y0, z0, current)
        k2 = self._derivatives(
            x0 + 0.5 * dt * k1[0],
            y0 + 0.5 * dt * k1[1],
            z0 + 0.5 * dt * k1[2],
            current,
        )
        k3 = self._derivatives(
            x0 + 0.5 * dt * k2[0],
            y0 + 0.5 * dt * k2[1],
            z0 + 0.5 * dt * k2[2],
            current,
        )
        k4 = self._derivatives(x0 + dt * k3[0], y0 + dt * k3[1], z0 + dt * k3[2], current)
        self._set_state(
            x0 + (dt / 6.0) * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]),
            y0 + (dt / 6.0) * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]),
            z0 + (dt / 6.0) * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]),
        )

    def step(self, current: float) -> int:
        if not isinstance(current, int | float) or not math.isfinite(float(current)):
            raise ValueError("current must be finite")
        current = float(current)
        x_prev = self.x
        if self.integrator == "rk4":
            self._step_rk4(current)
        else:
            self._step_euler(current)
        return 1 if (self.x >= self.x_threshold and x_prev < self.x_threshold) else 0

    def reset(self) -> None:
        self.x = -1.6
        self.y = -10.0
        self.z = 2.0
