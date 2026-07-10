# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Morris-Lecar 1981 — calcium-potassium oscillator

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
    ).MorrisLecarNeuron
    _HAS_RUST = True
except (ImportError, AttributeError):
    _EngineCls = None
    _HAS_RUST = False

_RUST_ENGINE_DEFAULTS: dict[str, float] = {
    "v": -60.0,
    "w": 0.0,
    "c_m": 20.0,
    "g_ca": 4.0,
    "g_k": 8.0,
    "g_l": 2.0,
    "e_ca": 120.0,
    "e_k": -84.0,
    "e_l": -60.0,
    "v1": -1.2,
    "v2": 18.0,
    "v3": 12.0,
    "v4": 17.4,
    "phi": 1.0 / 15.0,
    "dt": 0.1,
    "v_threshold": 0.0,
}

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
    - ``rk4`` is the maintained default fourth-order path over the Morris-Lecar ODEs
    - ``baseline_euler`` preserves the historical explicit-Euler path for comparison
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
    integrator: Literal["baseline_euler", "rk4", "rosenbrock"] = "rk4"

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
            raise ValueError(
                "Morris-Lecar conductance, scale, timestep, and rate parameters must be positive"
            )
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

    def _matches_rust_engine_contract(self) -> bool:
        """Return whether the instance matches the Rust engine default contract."""
        if self.integrator != "rk4":
            return False
        for name, expected in _RUST_ENGINE_DEFAULTS.items():
            if not math.isclose(float(getattr(self, name)), expected, rel_tol=0.0, abs_tol=0.0):
                return False
        return True

    def simulate(
        self, n_steps: int, current: float = 0.0, backend: str = "auto"
    ) -> tuple[npt.NDArray[np.float64], int]:
        """Advance ``n_steps`` updates, returning ``(v_trace, spikes)``.

        Rust path requires factory-default ``rk4`` parameters; parity is
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
                    "Rust MorrisLecar backend requested but sc_neurocore_engine is unavailable."
                )
            if not self._matches_rust_engine_contract():
                raise RuntimeError(
                    "Rust MorrisLecar backend requires factory-default rk4 "
                    "parameters and initial state."
                )
            trace, spikes, state = self._simulate_rust(n_steps, current)
        else:
            if backend == "rust":
                raise RuntimeError(
                    "Rust MorrisLecar backend requested but sc_neurocore_engine is unavailable."
                )
            trace, spikes, state = self._simulate_python(n_steps, current)
        self.v, self.w = state
        return trace, spikes

    def _simulate_python(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, tuple[float, float]]:
        trace = np.empty(n_steps, dtype=np.float64)
        spikes = 0
        for t in range(n_steps):
            spikes += self.step(current)
            trace[t] = self.v
        return trace, spikes, (self.v, self.w)

    def _simulate_rust(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, tuple[float, float]]:
        assert _EngineCls is not None
        neuron = _EngineCls()
        trace = np.empty(n_steps, dtype=np.float64)
        spikes = 0
        for t in range(n_steps):
            spikes += int(neuron.step(float(current)))
            st = neuron.get_state()
            trace[t] = float(st["v"])
        st = neuron.get_state()
        return trace, spikes, (float(st["v"]), float(st["w"]))

    def _rhs(self, _t: float, state: np.ndarray[Any, Any], current: float) -> np.ndarray[Any, Any]:
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
