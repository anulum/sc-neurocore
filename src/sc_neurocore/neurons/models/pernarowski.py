# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Pernarowski 1994 pancreatic beta-cell burster

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import ClassVar


@dataclass
class PernarowskiNeuron:
    """Pernarowski 1994 pancreatic beta-cell burster.

    Three coupled ODEs over ``(v, w, z)`` with one fast cubic state and
    two slower recovery/adaptation variables. The public implementation uses
    candidate-first RK4 integration and preserves continuous threshold-crossing
    semantics without an artificial reset during normal evolution.

    Reference: Pernarowski, M. (1994). SIAM J. Appl. Math. 54:814–832.
    """

    _FINITE_FIELDS: ClassVar[tuple[str, ...]] = (
        "v",
        "w",
        "z",
        "alpha",
        "beta",
        "v_threshold",
    )
    _POSITIVE_FIELDS: ClassVar[tuple[str, ...]] = ("eps1", "eps2", "gamma", "dt")

    v: float = -1.0
    w: float = 0.0
    z: float = 0.0
    alpha: float = 0.1
    beta: float = 0.5
    eps1: float = 0.1
    eps2: float = 0.001
    gamma: float = 0.5
    dt: float = 0.1
    v_threshold: float = 0.5

    def __post_init__(self) -> None:
        for name in self._FINITE_FIELDS:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a real finite scalar")
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            setattr(self, name, float(value))
        for name in self._POSITIVE_FIELDS:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a real positive scalar")
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            setattr(self, name, float(value))

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
        for name in self._POSITIVE_FIELDS:
            value = self._finite_float(name, getattr(self, name))
            if value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        return current

    def _derivatives(
        self, v: float, w: float, z: float, current: float
    ) -> tuple[float, float, float]:
        if not all(math.isfinite(value) for value in (v, w, z, current)):
            raise FloatingPointError("Pernarowski runtime state and current must be finite")
        try:
            dv = v - v**3 / 3.0 - w - z + current
            dw = self.eps1 * (v - self.gamma * w + self.alpha)
            dz = self.eps2 * (self.beta * (v + 0.7) - z)
        except OverflowError as exc:
            raise FloatingPointError("Pernarowski derivative overflow") from exc
        if not all(math.isfinite(value) for value in (dv, dw, dz)):
            raise FloatingPointError("Pernarowski derivative must be finite")
        return dv, dw, dz

    @staticmethod
    def _validate_candidate(v: float, w: float, z: float) -> None:
        if not all(math.isfinite(value) for value in (v, w, z)):
            raise FloatingPointError("Pernarowski RK4 candidate must be finite")

    def _rk4_candidate(self, current: float) -> tuple[float, float, float]:
        v0, w0, z0 = self.v, self.w, self.z
        dt = self.dt
        k1 = self._derivatives(v0, w0, z0, current)
        k2 = self._derivatives(
            v0 + 0.5 * dt * k1[0],
            w0 + 0.5 * dt * k1[1],
            z0 + 0.5 * dt * k1[2],
            current,
        )
        k3 = self._derivatives(
            v0 + 0.5 * dt * k2[0],
            w0 + 0.5 * dt * k2[1],
            z0 + 0.5 * dt * k2[2],
            current,
        )
        k4 = self._derivatives(v0 + dt * k3[0], w0 + dt * k3[1], z0 + dt * k3[2], current)
        candidate = (
            v0 + dt * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0,
            w0 + dt * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0,
            z0 + dt * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]) / 6.0,
        )
        self._validate_candidate(*candidate)
        return candidate

    def step(self, current: float = 0.0) -> int:
        current = self._validate_runtime_contract(current)
        v_prev = self.v
        v_new, w_new, z_new = self._rk4_candidate(current)
        self.v, self.w, self.z = v_new, w_new, z_new
        if self.v >= self.v_threshold and v_prev < self.v_threshold:
            return 1
        return 0

    def reset(self) -> None:
        self.v, self.w, self.z = -1.0, 0.0, 0.0
