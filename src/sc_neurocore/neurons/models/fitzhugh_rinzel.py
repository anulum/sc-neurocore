# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — FitzHugh 1976 / Rinzel 1987 — FHN + slow variable

from __future__ import annotations

import math
from dataclasses import dataclass


_STATE_NAMES = ("v", "w", "y")
_PARAMETER_NAMES = ("a", "b", "c", "d", "delta", "mu", "dt", "v_threshold")
_POSITIVE_PARAMETERS = ("b", "d", "delta", "mu", "dt")


@dataclass
class FitzHughRinzelNeuron:
    """FitzHugh-Rinzel three-state qualitative bursting model.

    dv/dt = v - v^3/3 - w + y + I
    dw/dt = delta * (a + v - b*w)
    dy/dt = mu * (c - v - d*y)

    Runtime integration uses RK4 over the published three-state ODE with
    current held constant for one step.
    """

    v: float = -1.0
    w: float = -0.5
    y: float = 0.0
    a: float = 0.7
    b: float = 0.8
    c: float = -0.775
    d: float = 1.0
    delta: float = 0.08
    mu: float = 0.0001
    dt: float = 0.1
    v_threshold: float = 1.0

    def __post_init__(self) -> None:
        self._validate_numeric_contract()

    @staticmethod
    def _finite_float(name: str, value: float) -> float:
        if isinstance(value, bool):
            raise ValueError(f"FitzHugh-Rinzel parameter {name} must be finite")
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"FitzHugh-Rinzel parameter {name} must be finite") from exc
        if not math.isfinite(result):
            raise ValueError(f"FitzHugh-Rinzel parameter {name} must be finite")
        return result

    def _numeric_fields(self) -> tuple[tuple[str, float], ...]:
        return tuple((name, getattr(self, name)) for name in (*_STATE_NAMES, *_PARAMETER_NAMES))

    def _validate_numeric_contract(self) -> None:
        for name, value in self._numeric_fields():
            setattr(self, name, self._finite_float(name, value))
        for name in _POSITIVE_PARAMETERS:
            if getattr(self, name) <= 0.0:
                raise ValueError(f"FitzHugh-Rinzel parameter {name} must be positive")

    def _derivatives(
        self, v: float, w: float, y: float, current: float
    ) -> tuple[float, float, float]:
        if not all(math.isfinite(value) for value in (v, w, y, current)):
            raise FloatingPointError("FitzHugh-Rinzel runtime state and current must be finite")
        try:
            dv = v - v**3 / 3.0 - w + y + current
            dw = self.delta * (self.a + v - self.b * w)
            dy = self.mu * (self.c - v - self.d * y)
        except OverflowError as exc:
            raise FloatingPointError("FitzHugh-Rinzel derivative overflow") from exc
        if not all(math.isfinite(value) for value in (dv, dw, dy)):
            raise FloatingPointError("FitzHugh-Rinzel derivative must be finite")
        return dv, dw, dy

    def _rk4_candidate(self, current: float) -> tuple[float, float, float]:
        v0, w0, y0, dt = self.v, self.w, self.y, self.dt
        k1 = self._derivatives(v0, w0, y0, current)
        k2 = self._derivatives(
            v0 + 0.5 * dt * k1[0],
            w0 + 0.5 * dt * k1[1],
            y0 + 0.5 * dt * k1[2],
            current,
        )
        k3 = self._derivatives(
            v0 + 0.5 * dt * k2[0],
            w0 + 0.5 * dt * k2[1],
            y0 + 0.5 * dt * k2[2],
            current,
        )
        k4 = self._derivatives(
            v0 + dt * k3[0],
            w0 + dt * k3[1],
            y0 + dt * k3[2],
            current,
        )
        return (
            v0 + dt * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0,
            w0 + dt * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0,
            y0 + dt * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]) / 6.0,
        )

    @staticmethod
    def _validate_candidate(v: float, w: float, y: float) -> tuple[float, float, float]:
        if not all(math.isfinite(value) for value in (v, w, y)):
            raise FloatingPointError("FitzHugh-Rinzel candidate state must be finite")
        return float(v), float(w), float(y)

    def step(self, current: float) -> int:
        """Advance the model by one RK4 step."""

        self._validate_numeric_contract()
        current = self._finite_float("current", current)
        v_prev = self.v
        self.v, self.w, self.y = self._validate_candidate(*self._rk4_candidate(current))
        return 1 if (self.v >= self.v_threshold and v_prev < self.v_threshold) else 0

    def reset(self) -> None:
        self.v, self.w, self.y = -1.0, -0.5, 0.0
