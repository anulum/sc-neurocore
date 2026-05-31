# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — McKean 1970 piecewise-linear FitzHugh-Nagumo caricature

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import ClassVar


@dataclass
class McKeanNeuron:
    """McKean 1970 piecewise-linear FitzHugh-Nagumo caricature.

    The model evolves the two-state ODE using candidate-first RK4 while
    preserving the three piecewise-linear voltage branches of McKean's
    analytically tractable Nagumo equation.

    Reference: McKean, H.P. (1970). Advances in Mathematics, 4:209-223.
    """

    _FINITE_FIELDS: ClassVar[tuple[str, ...]] = ("v", "w", "v_peak")
    _POSITIVE_FIELDS: ClassVar[tuple[str, ...]] = ("epsilon", "gamma", "dt")

    v: float = 0.0
    w: float = 0.0
    a: float = 0.25
    epsilon: float = 0.01
    gamma: float = 0.5
    dt: float = 0.1
    v_peak: float = 0.8

    def __post_init__(self) -> None:
        for name in self._FINITE_FIELDS:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a real finite scalar")
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            setattr(self, name, value)
        if isinstance(self.a, bool) or not isinstance(self.a, (int, float)):
            raise TypeError("a must be a real scalar in the open interval (0, 1)")
        self.a = float(self.a)
        if not math.isfinite(self.a) or not 0.0 < self.a < 1.0:
            raise ValueError("a must be finite and in the open interval (0, 1)")
        for name in self._POSITIVE_FIELDS:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a real positive scalar")
            value = float(value)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            setattr(self, name, value)

    @staticmethod
    def _finite_float(name: str, value: float) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a real finite scalar")
        value = float(value)
        if not math.isfinite(value):
            raise FloatingPointError(f"{name} must be finite")
        return value

    def _validate_runtime_contract(self, current: float) -> float:
        current = self._finite_float("current", current)
        for name in self._FINITE_FIELDS:
            self._finite_float(name, getattr(self, name))
        a = self._finite_float("a", self.a)
        if not 0.0 < a < 1.0:
            raise ValueError("a must be finite and in the open interval (0, 1)")
        for name in self._POSITIVE_FIELDS:
            value = self._finite_float(name, getattr(self, name))
            if value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        return current

    def _f(self, v: float) -> float:
        v = self._finite_float("v", v)
        mid1 = self.a / 2.0
        mid2 = (1.0 + self.a) / 2.0
        if v < mid1:
            return -v
        if v < mid2:
            return v - self.a
        return 1.0 - v

    def _derivatives(self, v: float, w: float, current: float) -> tuple[float, float]:
        if not all(math.isfinite(value) for value in (v, w, current)):
            raise FloatingPointError("McKean runtime state and current must be finite")
        dv = self._f(v) - w + current
        dw = self.epsilon * (v - self.gamma * w)
        if not math.isfinite(dv) or not math.isfinite(dw):
            raise FloatingPointError("McKean derivative must be finite")
        return dv, dw

    @staticmethod
    def _validate_candidate(v: float, w: float) -> None:
        if not math.isfinite(v) or not math.isfinite(w):
            raise FloatingPointError("McKean RK4 candidate must be finite")

    def _rk4_candidate(self, current: float) -> tuple[float, float]:
        v0, w0 = self.v, self.w
        dt = self.dt
        k1 = self._derivatives(v0, w0, current)
        k2 = self._derivatives(v0 + 0.5 * dt * k1[0], w0 + 0.5 * dt * k1[1], current)
        k3 = self._derivatives(v0 + 0.5 * dt * k2[0], w0 + 0.5 * dt * k2[1], current)
        k4 = self._derivatives(v0 + dt * k3[0], w0 + dt * k3[1], current)
        candidate = (
            v0 + dt * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0,
            w0 + dt * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0,
        )
        self._validate_candidate(*candidate)
        return candidate

    def step(self, current: float) -> int:
        current = self._validate_runtime_contract(current)
        v_prev = self.v
        self.v, self.w = self._rk4_candidate(current)
        return 1 if (self.v >= self.v_peak and v_prev < self.v_peak) else 0

    def reset(self) -> None:
        self.v = 0.0
        self.w = 0.0
