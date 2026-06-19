# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Adaptive Exponential Integrate-and-Fire. Brette &

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from sc_neurocore.solvers import RK4Solver, RosenbrockEuler


@dataclass
class AdExNeuron:
    """Adaptive Exponential Integrate-and-Fire. Brette & Gerstner 2005.

    dv/dt = -(v - v_rest)/tau + delta_T * exp((v - v_rh)/delta_T) / tau - w/C + I/C
    dw/dt = (a * (v - v_rest) - w) / tau_w
    if v >= v_threshold: v = v_reset, w += b

    Reference: Brette, R. & Gerstner, W. (2005). J. Neurophysiol. 94:3637–3642.

    Integrator options:
    - ``baseline_euler`` preserves the historical explicit-Euler path
    - ``rk4`` is an explicit higher-order alternative path
    - ``rosenbrock`` is a linearly implicit stiff-system path over the same
      AdEx ODEs
    """

    v: float = -65.0
    w: float = 0.0
    v_rest: float = -65.0
    v_reset: float = -68.0
    v_threshold: float = -50.0
    v_rh: float = -55.0
    delta_t: float = 2.0
    tau: float = 20.0
    tau_w: float = 100.0
    a: float = 0.5
    b: float = 7.0
    c_m: float = 200.0
    dt: float = 0.1
    integrator: Literal["baseline_euler", "rk4", "rosenbrock"] = "baseline_euler"

    def __post_init__(self) -> None:
        if self.integrator not in {"baseline_euler", "rk4", "rosenbrock"}:
            raise ValueError(f"Unsupported integrator for AdExNeuron: {self.integrator}")
        for field in ("v", "w", "v_rest", "v_reset", "v_threshold", "v_rh", "a", "b"):
            if not math.isfinite(getattr(self, field)):
                raise ValueError(f"{field} must be finite")
        for field in ("delta_t", "tau", "tau_w", "c_m", "dt"):
            value = getattr(self, field)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{field} must be finite and positive")

    def step(self, current: float) -> int:
        if not math.isfinite(current):
            raise ValueError("current must be finite")
        self._validate_runtime_state()
        if self.integrator == "baseline_euler":
            next_v, next_w = self._step_baseline_euler(current)
        elif self.integrator == "rk4":
            next_v, next_w = self._step_rk4(current)
        else:
            next_v, next_w = self._step_rosenbrock(current)

        self._validate_update(next_v, next_w)
        if next_v >= self.v_threshold:
            spike_w = next_w + self.b
            if not math.isfinite(spike_w):
                raise ValueError("spike adaptation update must remain finite")
            self.v = self.v_reset
            self.w = spike_w
            return 1
        self.v = next_v
        self.w = next_w
        return 0

    def _validate_runtime_state(self) -> None:
        if not math.isfinite(self.v):
            raise ValueError("runtime voltage state must be finite")
        if not math.isfinite(self.w):
            raise ValueError("runtime adaptation state must be finite")

    def _validate_update(self, next_v: float, next_w: float) -> None:
        if not math.isfinite(next_v) or not math.isfinite(next_w):
            raise ValueError("AdEx integrator update must remain finite")

    def _rhs(self, _t: float, state: np.ndarray[Any, Any], current: float) -> np.ndarray[Any, Any]:
        v = float(state[0])
        w = float(state[1])
        exp_term = self.delta_t * np.exp(np.clip((v - self.v_rh) / self.delta_t, -20.0, 20.0))
        dv = (-(v - self.v_rest) + exp_term) / self.tau + (-w + current) / self.c_m
        dw = (self.a * (v - self.v_rest) - w) / self.tau_w
        return np.array([dv, dw], dtype=np.float64)

    def _step_baseline_euler(self, current: float) -> tuple[float, float]:
        with np.errstate(over="ignore", invalid="ignore"):
            exp_term = self.delta_t * np.exp(
                np.clip((self.v - self.v_rh) / self.delta_t, -20.0, 20.0)
            )
            dv = (-(self.v - self.v_rest) + exp_term) / self.tau + (-self.w + current) / self.c_m
            dw = (self.a * (self.v - self.v_rest) - self.w) / self.tau_w
            return self.v + dv * self.dt, self.w + dw * self.dt

    def _step_rk4(self, current: float) -> tuple[float, float]:
        solver = RK4Solver()
        state = np.array([self.v, self.w], dtype=np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            state, _ = solver.step(
                lambda time, y: self._rhs(time, y, current),
                state,
                0.0,
                self.dt,
            )
        return float(state[0]), float(state[1])

    def _step_rosenbrock(self, current: float) -> tuple[float, float]:
        solver = RosenbrockEuler()
        state = np.array([self.v, self.w], dtype=np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            state, _ = solver.step(
                lambda time, y: self._rhs(time, y, current),
                state,
                0.0,
                self.dt,
            )
        return float(state[0]), float(state[1])

    def reset(self) -> None:
        self.v = self.v_rest
        self.w = 0.0
