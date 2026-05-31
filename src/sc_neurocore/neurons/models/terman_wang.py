# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Terman & Wang 1995 relaxation oscillator for LEGION

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import ClassVar


@dataclass
class TermanWangOscillator:
    """Terman & Wang 1995 relaxation oscillator for LEGION networks.

    The model evolves a fast excitatory variable ``v`` and slow recovery
    variable ``w`` under the published cubic/sigmoid ODE. Runtime integration
    uses candidate-first RK4 so invalid derivatives or candidates cannot poison
    state.

    Reference: Terman, D. & Wang, D.L. (1995). Neural Comput. 7:1035-1064.
    """

    _FINITE_FIELDS: ClassVar[tuple[str, ...]] = ("v", "w", "alpha", "rho", "v_peak")
    _POSITIVE_FIELDS: ClassVar[tuple[str, ...]] = ("beta", "epsilon", "dt")

    v: float = -1.5
    w: float = -0.5
    alpha: float = 3.0
    beta: float = 0.2
    epsilon: float = 0.02
    rho: float = 0.0
    dt: float = 0.05
    v_peak: float = 1.5

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

    def _derivatives(self, v: float, w: float, current: float) -> tuple[float, float]:
        if not all(math.isfinite(value) for value in (v, w, current)):
            raise FloatingPointError("Terman-Wang runtime state and current must be finite")
        try:
            f = 3.0 * v - v**3 + 2.0
            g = self.alpha * (1.0 + math.tanh(v / self.beta))
            dv = f - w + current + self.rho
            dw = self.epsilon * (g - w)
        except OverflowError as exc:
            raise FloatingPointError("Terman-Wang derivative overflow") from exc
        if not all(math.isfinite(value) for value in (dv, dw)):
            raise FloatingPointError("Terman-Wang derivative must be finite")
        return dv, dw

    @staticmethod
    def _validate_candidate(v: float, w: float) -> None:
        if not math.isfinite(v) or not math.isfinite(w):
            raise FloatingPointError("Terman-Wang RK4 candidate must be finite")

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
        self.v = -1.5
        self.w = -0.5
