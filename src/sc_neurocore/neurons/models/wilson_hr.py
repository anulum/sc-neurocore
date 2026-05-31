# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Wilson 1999 polynomial cortical model

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import ClassVar


@dataclass
class WilsonHRNeuron:
    """Wilson 1999 polynomial cortical model.

    dV/dt = -(17.81 + 47.71*V + 32.63*V^2)*(V - 0.55) - 26*R*(V + 0.92) + I
    dR/dt = (-R + 1.35*V + 1.03) / tau_R

    The maintained production path advances the coupled (V, R) state with
    candidate-first RK4 and commits only finite candidates. Spike detection is
    a threshold event followed by Wilson-HR hard voltage reset.
    """

    _FINITE_FIELDS: ClassVar[tuple[str, ...]] = ("v", "r", "v_peak")
    _POSITIVE_FIELDS: ClassVar[tuple[str, ...]] = ("tau_r", "dt")

    v: float = -0.7
    r: float = 0.1
    tau_r: float = 1.9
    v_peak: float = 0.4
    dt: float = 0.05

    def __post_init__(self) -> None:
        for name in self._FINITE_FIELDS:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a real finite scalar")
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            setattr(self, name, value)
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
        for name in self._POSITIVE_FIELDS:
            value = self._finite_float(name, getattr(self, name))
            if value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        return current

    @staticmethod
    def _poly(v: float) -> float:
        value = -(17.81 + 47.71 * v + 32.63 * v * v) * (v - 0.55)
        if not math.isfinite(value):
            raise FloatingPointError("Wilson-HR polynomial must be finite")
        return value

    def _derivatives(self, v: float, r: float, current: float) -> tuple[float, float]:
        if not all(math.isfinite(value) for value in (v, r, current)):
            raise FloatingPointError("Wilson-HR runtime state and current must be finite")
        poly = self._poly(v)
        syn = -26.0 * r * (v + 0.92)
        dv = poly + syn + current
        dr = (-r + 1.35 * v + 1.03) / self.tau_r
        if not math.isfinite(syn) or not math.isfinite(dv) or not math.isfinite(dr):
            raise FloatingPointError("Wilson-HR derivative must be finite")
        return dv, dr

    @staticmethod
    def _validate_candidate(v: float, r: float) -> None:
        if not math.isfinite(v) or not math.isfinite(r):
            raise FloatingPointError("Wilson-HR RK4 candidate must be finite")

    def _rk4_candidate(self, current: float) -> tuple[float, float]:
        v0, r0 = self.v, self.r
        dt = self.dt
        k1 = self._derivatives(v0, r0, current)
        k2 = self._derivatives(v0 + 0.5 * dt * k1[0], r0 + 0.5 * dt * k1[1], current)
        k3 = self._derivatives(v0 + 0.5 * dt * k2[0], r0 + 0.5 * dt * k2[1], current)
        k4 = self._derivatives(v0 + dt * k3[0], r0 + dt * k3[1], current)
        candidate = (
            v0 + dt * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0,
            r0 + dt * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0,
        )
        self._validate_candidate(*candidate)
        return candidate

    def step(self, current: float) -> int:
        current = self._validate_runtime_contract(current)
        next_v, next_r = self._rk4_candidate(current)
        self.v = next_v
        self.r = next_r
        if self.v >= self.v_peak:
            self.v = -0.7
            return 1
        return 0

    def reset(self) -> None:
        self.v = -0.7
        self.r = 0.1
