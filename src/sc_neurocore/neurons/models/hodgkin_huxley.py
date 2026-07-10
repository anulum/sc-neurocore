# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Hodgkin-Huxley 1952 — 4-ODE ion channel model

from __future__ import annotations

import importlib as _importlib
import math
from dataclasses import dataclass
from typing import Any, Literal, Optional

import numpy as np
import numpy.typing as npt

from sc_neurocore.solvers import RK4Solver, RosenbrockEuler

try:
    _EngineCls: Optional[type[Any]] = _importlib.import_module(
        "sc_neurocore_engine"
    ).HodgkinHuxleyNeuron
    _HAS_RUST = True
except (ImportError, AttributeError):
    _EngineCls = None
    _HAS_RUST = False

_RUST_ENGINE_DEFAULTS: dict[str, float] = {
    "v": -65.0,
    "m": 0.05,
    "h": 0.6,
    "n": 0.32,
    "c_m": 1.0,
    "g_na": 120.0,
    "g_k": 36.0,
    "g_l": 0.3,
    "e_na": 50.0,
    "e_k": -77.0,
    "e_l": -54.4,
    "dt": 0.01,
    "v_threshold": 0.0,
}


@dataclass
class HodgkinHuxleyNeuron:
    """Hodgkin-Huxley 1952 — 4-ODE ion channel model.

    C_m dv/dt = -g_Na m³h(v-E_Na) - g_K n⁴(v-E_K) - g_L(v-E_L) + I
    dm/dt = α_m(1-m) - β_m·m
    dh/dt = α_h(1-h) - β_h·h
    dn/dt = α_n(1-n) - β_n·n

    Reference: Hodgkin, A.L. & Huxley, A.F. (1952). J. Physiol. 117:500–544.

    Integrator options:
    - ``baseline_euler`` preserves the historical explicit-Euler sub-step path
    - ``rk4`` is an explicit higher-order alternative path over the same
      sub-step schedule
    - ``rosenbrock`` is a linearly implicit stiff-system path over the same
      Hodgkin-Huxley ODEs
    """

    v: float = -65.0
    m: float = 0.05
    h: float = 0.6
    n: float = 0.32
    c_m: float = 1.0
    g_na: float = 120.0
    g_k: float = 36.0
    g_l: float = 0.3
    e_na: float = 50.0
    e_k: float = -77.0
    e_l: float = -54.4
    dt: float = 0.01
    v_threshold: float = 0.0
    integrator: Literal["baseline_euler", "rk4", "rosenbrock"] = "baseline_euler"

    def __post_init__(self) -> None:
        if self.integrator not in {"baseline_euler", "rk4", "rosenbrock"}:
            raise ValueError(f"Unsupported integrator for HodgkinHuxleyNeuron: {self.integrator}")

    def _alpha_m(self, v: float) -> float:
        d = v + 40.0
        if abs(d) < 1e-7:
            return 1.0
        return float(0.1 * d / (1.0 - np.exp(-d / 10.0)))

    def _beta_m(self, v: float) -> float:
        return float(4.0 * np.exp(-(v + 65.0) / 18.0))

    def _alpha_h(self, v: float) -> float:
        return float(0.07 * np.exp(-(v + 65.0) / 20.0))

    def _beta_h(self, v: float) -> float:
        return float(1.0 / (1.0 + np.exp(-(v + 35.0) / 10.0)))

    def _alpha_n(self, v: float) -> float:
        d = v + 55.0
        if abs(d) < 1e-7:
            return 0.1
        return float(0.01 * d / (1.0 - np.exp(-d / 10.0)))

    def _beta_n(self, v: float) -> float:
        return float(0.125 * np.exp(-(v + 65.0) / 80.0))

    def step(self, current: float) -> int:
        v_prev = self.v
        if self.integrator == "baseline_euler":
            self._step_baseline_euler(current)
        elif self.integrator == "rk4":
            self._step_rk4(current)
        else:
            self._step_rosenbrock(current)
        return 1 if (self.v >= self.v_threshold and v_prev < self.v_threshold) else 0

    def _matches_rust_engine_contract(self) -> bool:
        """Return whether the instance matches the Rust engine default contract."""
        if self.integrator != "baseline_euler":
            return False
        for name, expected in _RUST_ENGINE_DEFAULTS.items():
            if float(getattr(self, name)) != expected:
                return False
        return True

    def simulate(
        self, n_steps: int, current: float = 0.0, backend: str = "auto"
    ) -> tuple[npt.NDArray[np.float64], int]:
        """Advance ``n_steps`` updates, returning ``(v_trace, spikes)``.

        Rust path requires factory-default ``baseline_euler``; parity is
        ULP-bounded against the pure-Python path.
        """
        if n_steps < 0:
            raise ValueError("n_steps must be non-negative")
        if backend not in ("auto", "python", "rust"):
            raise ValueError(f"backend must be auto/python/rust, got {backend!r}")
        if not math.isfinite(current):
            raise ValueError("current must be finite")
        prefer_rust = backend == "rust" or (
            backend == "auto" and _HAS_RUST and self._matches_rust_engine_contract()
        )
        if prefer_rust:
            if not _HAS_RUST or _EngineCls is None:
                raise RuntimeError(
                    "Rust HodgkinHuxley backend requested but sc_neurocore_engine is unavailable."
                )
            if not self._matches_rust_engine_contract():
                raise RuntimeError(
                    "Rust HodgkinHuxley backend requires factory-default "
                    "baseline_euler parameters and initial state."
                )
            trace, spikes, state = self._simulate_rust(n_steps, current)
        else:
            if backend == "rust":
                raise RuntimeError(
                    "Rust HodgkinHuxley backend requested but sc_neurocore_engine is unavailable."
                )
            trace, spikes, state = self._simulate_python(n_steps, current)
        self.v, self.m, self.h, self.n = state
        return trace, spikes

    def _simulate_python(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, tuple[float, float, float, float]]:
        trace = np.empty(n_steps, dtype=np.float64)
        spikes = 0
        for t in range(n_steps):
            spikes += self.step(current)
            trace[t] = self.v
        return trace, spikes, (self.v, self.m, self.h, self.n)

    def _simulate_rust(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, tuple[float, float, float, float]]:
        assert _EngineCls is not None
        neuron = _EngineCls()
        trace = np.empty(n_steps, dtype=np.float64)
        spikes = 0
        for t in range(n_steps):
            spikes += int(neuron.step(float(current)))
            st = neuron.get_state()
            trace[t] = float(st["v"])
        st = neuron.get_state()
        return (
            trace,
            spikes,
            (float(st["v"]), float(st["m"]), float(st["h"]), float(st["n"])),
        )

    def _rhs(self, _t: float, state: np.ndarray[Any, Any], current: float) -> np.ndarray[Any, Any]:
        v, m, h, n = (float(value) for value in state)
        am, bm = self._alpha_m(v), self._beta_m(v)
        ah, bh = self._alpha_h(v), self._beta_h(v)
        an, bn = self._alpha_n(v), self._beta_n(v)

        dm = am * (1.0 - m) - bm * m
        dh = ah * (1.0 - h) - bh * h
        dn = an * (1.0 - n) - bn * n

        i_na = self.g_na * m**3 * h * (v - self.e_na)
        i_k = self.g_k * n**4 * (v - self.e_k)
        i_l = self.g_l * (v - self.e_l)
        dv = (-i_na - i_k - i_l + current) / self.c_m
        return np.array([dv, dm, dh, dn], dtype=np.float64)

    def _step_baseline_euler(self, current: float) -> None:
        for _ in range(round(1.0 / self.dt)):
            am, bm = self._alpha_m(self.v), self._beta_m(self.v)
            ah, bh = self._alpha_h(self.v), self._beta_h(self.v)
            an, bn = self._alpha_n(self.v), self._beta_n(self.v)

            self.m += (am * (1 - self.m) - bm * self.m) * self.dt
            self.h += (ah * (1 - self.h) - bh * self.h) * self.dt
            self.n += (an * (1 - self.n) - bn * self.n) * self.dt

            i_na = self.g_na * self.m**3 * self.h * (self.v - self.e_na)
            i_k = self.g_k * self.n**4 * (self.v - self.e_k)
            i_l = self.g_l * (self.v - self.e_l)

            self.v += (-i_na - i_k - i_l + current) / self.c_m * self.dt

    def _step_rk4(self, current: float) -> None:
        solver = RK4Solver()
        state = np.array([self.v, self.m, self.h, self.n], dtype=np.float64)
        t = 0.0
        for _ in range(round(1.0 / self.dt)):
            state, dt_used = solver.step(
                lambda time, y: self._rhs(time, y, current),
                state,
                t,
                self.dt,
            )
            t += dt_used
        self.v, self.m, self.h, self.n = (float(value) for value in state)

    def _step_rosenbrock(self, current: float) -> None:
        solver = RosenbrockEuler()
        state = np.array([self.v, self.m, self.h, self.n], dtype=np.float64)
        t = 0.0
        for _ in range(round(1.0 / self.dt)):
            state, dt_used = solver.step(
                lambda time, y: self._rhs(time, y, current),
                state,
                t,
                self.dt,
            )
            t += dt_used
        self.v, self.m, self.h, self.n = (float(value) for value in state)

    def reset(self) -> None:
        self.v = -65.0
        self.m = 0.05
        self.h = 0.6
        self.n = 0.32
