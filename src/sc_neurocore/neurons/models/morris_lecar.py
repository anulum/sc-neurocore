# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Morris-Lecar 1981 — calcium-potassium oscillator

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import math

import numpy as np

from sc_neurocore.solvers import RK4Solver, RosenbrockEuler


_STATE_NAMES = ("v", "w")
_PARAM_NAMES = (
    "c_m",
    "g_ca",
    "g_k",
    "g_l",
    "e_ca",
    "e_k",
    "e_l",
    "v1",
    "v2",
    "v3",
    "v4",
    "phi",
    "dt",
    "v_threshold",
)
_STRICTLY_POSITIVE_PARAMS = ("c_m", "g_ca", "g_k", "g_l", "v2", "v4", "phi", "dt")


@dataclass
class MorrisLecarNeuron:
    """Morris-Lecar 1981 — calcium-potassium oscillator.

    C dv/dt = -g_Ca m_∞(v)(v-E_Ca) - g_K w(v-E_K) - g_L(v-E_L) + I
    dw/dt = λ(v)(w_∞(v) - w)

    Reference: Morris, C. & Lecar, H. (1981). Biophys. J. 35:193–213.

    Integrator options:
    - ``baseline_euler`` preserves the historical explicit-Euler path
    - ``rk4`` is an explicit fourth-order path over the same Morris-Lecar ODEs
    - ``rosenbrock`` is a linearly implicit stiff-system path over the same
      conductance equations
    """

    v: float = -60.0
    w: float = 0.0
    c_m: float = 20.0
    g_ca: float = 4.0
    g_k: float = 8.0
    g_l: float = 2.0
    e_ca: float = 120.0
    e_k: float = -84.0
    e_l: float = -60.0
    v1: float = -1.2
    v2: float = 18.0
    v3: float = 12.0
    v4: float = 17.4
    phi: float = 1.0 / 15.0
    dt: float = 0.1
    v_threshold: float = 0.0
    integrator: Literal["baseline_euler", "rk4", "rosenbrock"] = "baseline_euler"

    def __post_init__(self) -> None:
        if self.integrator not in {"baseline_euler", "rk4", "rosenbrock"}:
            raise ValueError(f"Unsupported integrator for MorrisLecarNeuron: {self.integrator}")
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
        if not 0.0 <= self.w <= 1.0:
            raise ValueError("w must remain in [0, 1]")

    def _validate_runtime_configuration(self) -> None:
        if not all(math.isfinite(getattr(self, name)) for name in (*_STATE_NAMES, *_PARAM_NAMES)):
            raise ValueError("Morris-Lecar state and parameters must be finite")
        if any(getattr(self, name) <= 0.0 for name in _STRICTLY_POSITIVE_PARAMS):
            raise ValueError("Morris-Lecar conductance, scale, timestep, and rate parameters must be positive")
        if not 0.0 <= self.w <= 1.0:
            raise ValueError("w must remain in [0, 1]")

    def _m_inf(self, v: float) -> float:
        if not math.isfinite(v):
            raise FloatingPointError("Morris-Lecar voltage became non-finite")
        out = 0.5 * (1.0 + math.tanh((v - self.v1) / self.v2))
        if not math.isfinite(out):
            raise FloatingPointError("Morris-Lecar calcium activation became non-finite")
        return out

    def _w_inf(self, v: float) -> float:
        if not math.isfinite(v):
            raise FloatingPointError("Morris-Lecar voltage became non-finite")
        out = 0.5 * (1.0 + math.tanh((v - self.v3) / self.v4))
        if not math.isfinite(out):
            raise FloatingPointError("Morris-Lecar potassium activation became non-finite")
        return out

    def _lam(self, v: float) -> float:
        if not math.isfinite(v):
            raise FloatingPointError("Morris-Lecar voltage became non-finite")
        try:
            out = self.phi * math.cosh((v - self.v3) / (2.0 * self.v4))
        except OverflowError as exc:
            raise FloatingPointError("Morris-Lecar potassium rate overflowed") from exc
        if not math.isfinite(out):
            raise FloatingPointError("Morris-Lecar potassium rate became non-finite")
        return out

    @staticmethod
    def _validate_state(v: float, w: float) -> tuple[float, float]:
        if not (math.isfinite(v) and math.isfinite(w)):
            raise FloatingPointError("Morris-Lecar state became non-finite")
        if not 0.0 <= w <= 1.0:
            raise FloatingPointError("Morris-Lecar potassium activation left [0, 1]")
        return float(v), float(w)

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

    def _rhs(self, _t: float, state: np.ndarray, current: float) -> np.ndarray:
        v = float(state[0])
        w = float(state[1])
        m_inf = self._m_inf(v)
        w_inf = self._w_inf(v)
        lam = self._lam(v)
        i_ca = self.g_ca * m_inf * (v - self.e_ca)
        i_k = self.g_k * w * (v - self.e_k)
        i_l = self.g_l * (v - self.e_l)
        dv = (-i_ca - i_k - i_l + current) / self.c_m
        dw = lam * (w_inf - w)
        out = np.array([dv, dw], dtype=np.float64)
        if not np.all(np.isfinite(out)):
            raise FloatingPointError("Morris-Lecar derivative became non-finite")
        return out

    def _step_baseline_euler(self, current: float) -> None:
        m_inf = self._m_inf(self.v)
        w_inf = self._w_inf(self.v)
        lam = self._lam(self.v)
        i_ca = self.g_ca * m_inf * (self.v - self.e_ca)
        i_k = self.g_k * self.w * (self.v - self.e_k)
        i_l = self.g_l * (self.v - self.e_l)
        new_v = self.v + (-i_ca - i_k - i_l + current) / self.c_m * self.dt
        new_w = self.w + lam * (w_inf - self.w) * self.dt
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
        self.v = -60.0
        self.w = 0.0
