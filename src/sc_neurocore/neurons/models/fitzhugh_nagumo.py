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

from sc_neurocore.solvers import RK4Solver, RosenbrockEuler


_STATE_NAMES = ("v", "w")
_PARAM_NAMES = ("a", "b", "epsilon", "dt", "v_threshold")
_STRICTLY_POSITIVE_PARAMS = ("b", "epsilon", "dt")


@dataclass
class FitzHughNagumoNeuron:
    """FitzHugh-Nagumo 1961 — 2D qualitative spike model.

    dv/dt = v - v³/3 - w + I
    dw/dt = ε(v + a - bw)

    Reference: FitzHugh, R. (1961). Biophys. J. 1:445–466.

    Integrator options:
    - ``baseline_euler`` preserves the historical explicit-Euler path
    - ``rk4`` is an explicit fourth-order path over the same two-state ODE
    - ``rosenbrock`` is a linearly implicit path for stiff slow-fast regimes
    """

    v: float = -1.0
    w: float = -0.5
    a: float = 0.7
    b: float = 0.8
    epsilon: float = 0.08
    dt: float = 0.1
    v_threshold: float = 1.0
    integrator: Literal["baseline_euler", "rk4", "rosenbrock"] = "baseline_euler"

    def __post_init__(self) -> None:
        if self.integrator not in {"baseline_euler", "rk4", "rosenbrock"}:
            raise ValueError(f"Unsupported integrator for FitzHughNagumoNeuron: {self.integrator}")
        self._validate_configuration()

    def _validate_configuration(self) -> None:
        for name in (*_STATE_NAMES, *_PARAM_NAMES):
            value = getattr(self, name)
            if not isinstance(value, int | float) or not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
            setattr(self, name, float(value))
        for name in _STRICTLY_POSITIVE_PARAMS:
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")

    def _validate_runtime_configuration(self) -> None:
        if not (
            math.isfinite(self.v)
            and math.isfinite(self.w)
            and math.isfinite(self.a)
            and math.isfinite(self.b)
            and math.isfinite(self.epsilon)
            and math.isfinite(self.dt)
            and math.isfinite(self.v_threshold)
        ):
            raise ValueError("FitzHugh-Nagumo state and parameters must be finite")
        if self.b <= 0.0 or self.epsilon <= 0.0 or self.dt <= 0.0:
            raise ValueError("b, epsilon, and dt must be positive")

    def step(self, current: float) -> int:
        if not isinstance(current, int | float) or not math.isfinite(float(current)):
            raise ValueError("current must be finite")
        current = float(current)
        self._validate_runtime_configuration()
        v_prev = self.v
        if self.integrator == "baseline_euler":
            self._step_baseline_euler(current)
        elif self.integrator == "rk4":
            self._step_rk4(current)
        else:
            self._step_rosenbrock(current)
        return 1 if (self.v >= self.v_threshold and v_prev < self.v_threshold) else 0

    @staticmethod
    def _validate_state(v: float, w: float) -> tuple[float, float]:
        if not (math.isfinite(v) and math.isfinite(w)):
            raise FloatingPointError("FitzHugh-Nagumo state became non-finite")
        return float(v), float(w)

    def _rhs(self, _t: float, state: np.ndarray, current: float) -> np.ndarray:
        v = float(state[0])
        w = float(state[1])
        if not (math.isfinite(v) and math.isfinite(w) and math.isfinite(current)):
            raise FloatingPointError("FitzHugh-Nagumo derivative input became non-finite")
        try:
            dv = v - v**3 / 3.0 - w + current
            dw = self.epsilon * (v + self.a - self.b * w)
        except OverflowError as exc:
            raise FloatingPointError("FitzHugh-Nagumo derivative overflowed") from exc
        out = np.array([dv, dw], dtype=np.float64)
        if not np.all(np.isfinite(out)):
            raise FloatingPointError("FitzHugh-Nagumo derivative became non-finite")
        return out

    def _step_baseline_euler(self, current: float) -> None:
        dv, dw = self._rhs(0.0, np.array([self.v, self.w], dtype=np.float64), current)
        new_v = self.v + dv * self.dt
        new_w = self.w + dw * self.dt
        self.v, self.w = self._validate_state(new_v, new_w)

    def _step_rk4(self, current: float) -> None:
        solver = RK4Solver()
        state = np.array([self.v, self.w], dtype=np.float64)
        state, _ = solver.step(
            lambda time, y: self._rhs(time, y, current),
            state,
            0.0,
            self.dt,
        )
        self.v, self.w = self._validate_state(float(state[0]), float(state[1]))

    def _step_rosenbrock(self, current: float) -> None:
        solver = RosenbrockEuler()
        state = np.array([self.v, self.w], dtype=np.float64)
        state, _ = solver.step(
            lambda time, y: self._rhs(time, y, current),
            state,
            0.0,
            self.dt,
        )
        self.v, self.w = self._validate_state(float(state[0]), float(state[1]))

    def reset(self) -> None:
        self.v = -1.0
        self.w = -0.5
