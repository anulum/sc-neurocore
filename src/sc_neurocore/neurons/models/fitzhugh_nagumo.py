# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — FitzHugh-Nagumo 1961 — 2D qualitative spike model

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import math

import numpy as np

from sc_neurocore.solvers import RosenbrockEuler


_STATE_NAMES = ("v", "w")
_PARAM_NAMES = ("a", "b", "epsilon", "dt", "v_threshold")
_STRICTLY_POSITIVE_PARAMS = ("b", "epsilon", "dt")


@dataclass
class FitzHughNagumoNeuron:
    """FitzHugh-Nagumo 1961 two-state excitable-system model.

    dv/dt = v - v^3 / 3 - w + I
    dw/dt = epsilon * (v + a - b*w)

    The production default is RK4 over the published two-state ODE. The
    historical explicit-Euler path remains available only through the explicit
    ``baseline_euler`` integrator option for compatibility experiments.
    """

    v: float = -1.0
    w: float = -0.5
    a: float = 0.7
    b: float = 0.8
    epsilon: float = 0.08
    dt: float = 0.1
    v_threshold: float = 1.0
    integrator: Literal["baseline_euler", "rk4", "rosenbrock"] = "rk4"

    def __post_init__(self) -> None:
        if self.integrator not in {"baseline_euler", "rk4", "rosenbrock"}:
            raise ValueError(f"Unsupported integrator for FitzHughNagumoNeuron: {self.integrator}")
        self._validate_configuration()

    @staticmethod
    def _finite_float(name: str, value: float) -> float:
        if isinstance(value, bool):
            raise ValueError(f"{name} must be finite")
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be finite") from exc
        if not math.isfinite(result):
            raise ValueError(f"{name} must be finite")
        return result

    def _validate_configuration(self) -> None:
        for name in (*_STATE_NAMES, *_PARAM_NAMES):
            setattr(self, name, self._finite_float(name, getattr(self, name)))
        for name in _STRICTLY_POSITIVE_PARAMS:
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")

    def _validate_runtime_configuration(self) -> None:
        for name in (*_STATE_NAMES, *_PARAM_NAMES):
            self._finite_float(name, getattr(self, name))
        for name in _STRICTLY_POSITIVE_PARAMS:
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")

    def step(self, current: float) -> int:
        current = self._finite_float("current", current)
        self._validate_runtime_configuration()
        v_prev = self.v
        if self.integrator == "baseline_euler":
            candidate = self._euler_candidate(current)
        elif self.integrator == "rk4":
            candidate = self._rk4_candidate(current)
        else:
            candidate = self._rosenbrock_candidate(current)
        self.v, self.w = self._validate_candidate(*candidate)
        return 1 if (self.v >= self.v_threshold and v_prev < self.v_threshold) else 0

    @staticmethod
    def _validate_candidate(v: float, w: float) -> tuple[float, float]:
        if not math.isfinite(v):
            raise FloatingPointError("FitzHugh-Nagumo v candidate became non-finite")
        if not math.isfinite(w):
            raise FloatingPointError("FitzHugh-Nagumo w candidate became non-finite")
        return float(v), float(w)

    def _rhs_tuple(self, v: float, w: float, current: float) -> tuple[float, float]:
        if not (math.isfinite(v) and math.isfinite(w) and math.isfinite(current)):
            raise FloatingPointError("FitzHugh-Nagumo derivative input became non-finite")
        try:
            dv = v - v**3 / 3.0 - w + current
            dw = self.epsilon * (v + self.a - self.b * w)
        except OverflowError as exc:
            raise FloatingPointError("FitzHugh-Nagumo derivative overflowed") from exc
        if not (math.isfinite(dv) and math.isfinite(dw)):
            raise FloatingPointError("FitzHugh-Nagumo derivative became non-finite")
        return dv, dw

    def _rhs(self, _t: float, state: np.ndarray, current: float) -> np.ndarray:
        dv, dw = self._rhs_tuple(float(state[0]), float(state[1]), current)
        return np.array([dv, dw], dtype=np.float64)

    def _euler_candidate(self, current: float) -> tuple[float, float]:
        dv, dw = self._rhs_tuple(self.v, self.w, current)
        return self.v + dv * self.dt, self.w + dw * self.dt

    def _rk4_candidate(self, current: float) -> tuple[float, float]:
        v0, w0, dt = self.v, self.w, self.dt
        k1v, k1w = self._rhs_tuple(v0, w0, current)
        k2v, k2w = self._rhs_tuple(v0 + 0.5 * dt * k1v, w0 + 0.5 * dt * k1w, current)
        k3v, k3w = self._rhs_tuple(v0 + 0.5 * dt * k2v, w0 + 0.5 * dt * k2w, current)
        k4v, k4w = self._rhs_tuple(v0 + dt * k3v, w0 + dt * k3w, current)
        return (
            v0 + dt * (k1v + 2.0 * k2v + 2.0 * k3v + k4v) / 6.0,
            w0 + dt * (k1w + 2.0 * k2w + 2.0 * k3w + k4w) / 6.0,
        )

    def _rosenbrock_candidate(self, current: float) -> tuple[float, float]:
        solver = RosenbrockEuler()
        state = np.array([self.v, self.w], dtype=np.float64)
        state, _ = solver.step(
            lambda time, y: self._rhs(time, y, current),
            state,
            0.0,
            self.dt,
        )
        return float(state[0]), float(state[1])

    def reset(self) -> None:
        self.v = -1.0
        self.w = -0.5
