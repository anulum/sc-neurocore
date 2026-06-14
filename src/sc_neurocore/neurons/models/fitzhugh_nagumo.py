# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — FitzHugh-Nagumo 1961 — 2D qualitative spike model

from __future__ import annotations

import importlib as _importlib
import os as _os
from dataclasses import dataclass
from typing import Callable, Literal, Optional

import math

import numpy as np
import numpy.typing as npt

from sc_neurocore.solvers import RosenbrockEuler


_STATE_NAMES = ("v", "w")
_PARAM_NAMES = ("a", "b", "epsilon", "dt", "v_threshold")
_STRICTLY_POSITIVE_PARAMS = ("b", "epsilon", "dt")

# ───────────────────────── backend detection ─────────────────────────
#
# `step` runs one RK4 update; an N-step `simulate` is a sequential recurrence
# (each step depends on the previous) that does not vectorise, so a compiled
# inner loop genuinely beats Python. The polyglot chain accelerates the RK4
# integrator (the production default). The FitzHugh-Nagumo right-hand side is
# exact arithmetic (a cube, additions and multiplications — no transcendental
# functions), and the system is two-dimensional, so by Poincaré-Bendixson it
# cannot be chaotic: Rust, Julia and Go reproduce the NumPy trace bit-for-bit
# and Mojo agrees to a small, non-amplifying ULP bound.

_RustSimulate = Callable[..., "tuple[list[float], int, float, float]"]


def _load_rust_simulate() -> _RustSimulate:
    engine = _importlib.import_module("sc_neurocore_engine")
    return engine.py_fitzhugh_nagumo_simulate  # type: ignore[no-any-return]


try:
    _rust_simulate: Optional[_RustSimulate] = _load_rust_simulate()
    _HAS_RUST = True
except (ImportError, AttributeError):
    _rust_simulate = None
    _HAS_RUST = False

_julia_module = None
_HAS_JULIA = False
_go_lib = None
_HAS_GO = False
_mojo_lib = None
_HAS_MOJO = False

_ACCEL_ROOT = _os.path.join(_os.path.dirname(__file__), "..", "..", "accel")


def _ensure_julia_loaded() -> bool:
    global _julia_module, _HAS_JULIA
    if _julia_module is not None:
        return True
    import importlib.util as importlib_util

    if importlib_util.find_spec("juliacall") is None:
        return False
    jl_path = _os.path.abspath(_os.path.join(_ACCEL_ROOT, "julia", "neurons", "fitzhugh_nagumo.jl"))
    if not _os.path.isfile(jl_path):
        return False
    juliacall = _importlib.import_module("juliacall")
    jl = juliacall.Main
    jl.include(jl_path)
    _julia_module = jl.FitzHughNagumoAccel
    _HAS_JULIA = True
    return True


def _ensure_go_loaded() -> bool:
    global _go_lib, _HAS_GO
    if _go_lib is not None:
        return True
    import ctypes

    so_path = _os.path.abspath(
        _os.path.join(_ACCEL_ROOT, "go", "neurons", "fitzhugh_nagumo", "libfhn.so")
    )
    if not _os.path.isfile(so_path):
        return False
    try:
        lib = ctypes.CDLL(so_path)
    except OSError:
        return False
    fn = getattr(lib, "fitzhugh_nagumo_simulate_c", None)
    if fn is None:
        return False
    fn.argtypes = [ctypes.c_double] * 7 + [
        ctypes.c_int,
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_double),
    ]
    fn.restype = ctypes.c_longlong
    _go_lib = lib
    _HAS_GO = True
    return True


def _ensure_mojo_loaded() -> bool:
    global _mojo_lib, _HAS_MOJO
    if _mojo_lib is not None:
        return True
    import ctypes

    so_path = _os.path.abspath(_os.path.join(_ACCEL_ROOT, "mojo", "neurons", "libfhn.so"))
    if not _os.path.isfile(so_path):
        return False
    try:
        lib = ctypes.CDLL(so_path)
    except OSError:
        return False
    fn = getattr(lib, "fitzhugh_nagumo_simulate_c", None)
    if fn is None:
        return False
    # 7 float params + n_steps + current + trace addr; returns spikes.
    fn.argtypes = [ctypes.c_double] * 7 + [ctypes.c_int64, ctypes.c_double, ctypes.c_int64]
    fn.restype = ctypes.c_int64
    _mojo_lib = lib
    _HAS_MOJO = True
    return True


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

    def simulate(
        self, n_steps: int, current: float = 0.0, backend: str = "auto"
    ) -> tuple[npt.NDArray[np.float64], int]:
        """Advance ``n_steps`` RK4 steps from the current state, returning ``(trace, spikes)``.

        ``trace[t]`` is the membrane variable ``v`` after step ``t``; ``spikes``
        counts upward crossings of ``v_threshold``. The instance state
        ``(v, w)`` is advanced to the final step.

        The compiled backends implement the **RK4** integrator (the production
        default); this method therefore requires ``integrator == "rk4"``. Rust,
        Julia and Go reproduce the pure-NumPy reference bit-for-bit (the RHS is
        exact arithmetic); Mojo agrees to a documented, non-amplifying ULP bound.
        """
        if n_steps < 0:
            raise ValueError("n_steps must be non-negative")
        current = self._finite_float("current", current)
        if self.integrator != "rk4":
            raise ValueError(
                "simulate() accelerates the RK4 integrator only; this neuron uses "
                f"integrator={self.integrator!r}. Call step() in a loop, or set integrator='rk4'."
            )
        if backend not in ("auto", "rust", "julia", "go", "mojo", "python"):
            raise ValueError(f"backend must be auto/rust/julia/go/mojo/python, got {backend!r}")
        self._validate_runtime_configuration()

        if backend == "rust" and not _HAS_RUST:
            raise RuntimeError("Rust FitzHugh-Nagumo backend requested but the engine lacks it.")
        if backend == "julia" and not _ensure_julia_loaded():
            raise RuntimeError("Julia FitzHugh-Nagumo backend requested but it is unavailable.")
        if backend == "go" and not _ensure_go_loaded():
            raise RuntimeError(
                "Go FitzHugh-Nagumo backend requested but libfhn.so is not built; run "
                "`cd src/sc_neurocore/accel/go/neurons/fitzhugh_nagumo && go build "
                "-buildmode=c-shared -o libfhn.so fitzhugh_nagumo.go`."
            )
        if backend == "mojo" and not _ensure_mojo_loaded():
            raise RuntimeError(
                "Mojo FitzHugh-Nagumo backend requested but libfhn.so is not built; run "
                "`cd src/sc_neurocore/accel/mojo/neurons && mojo build --emit shared-lib "
                "-o libfhn.so fitzhugh_nagumo.mojo`."
            )

        if backend == "rust" or (backend == "auto" and _HAS_RUST):
            trace, spikes, vf, wf = self._simulate_rust(n_steps, current)
        elif backend == "julia":
            trace, spikes, vf, wf = self._simulate_julia(n_steps, current)
        elif backend == "go":
            trace, spikes, vf, wf = self._simulate_go(n_steps, current)
        elif backend == "mojo":
            trace, spikes, vf, wf = self._simulate_mojo(n_steps, current)
        else:
            trace, spikes, vf, wf = self._simulate_python(n_steps, current)
        self.v, self.w = vf, wf
        return trace, spikes

    def _simulate_python(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, float, float]:
        trace = np.empty(n_steps, dtype=np.float64)
        v, w = self.v, self.w
        a, b, eps, dt, thr = self.a, self.b, self.epsilon, self.dt, self.v_threshold
        spikes = 0
        for t in range(n_steps):
            v_prev = v
            k1v = v - v * v * v / 3.0 - w + current
            k1w = eps * (v + a - b * w)
            v2 = v + 0.5 * dt * k1v
            w2 = w + 0.5 * dt * k1w
            k2v = v2 - v2 * v2 * v2 / 3.0 - w2 + current
            k2w = eps * (v2 + a - b * w2)
            v3 = v + 0.5 * dt * k2v
            w3 = w + 0.5 * dt * k2w
            k3v = v3 - v3 * v3 * v3 / 3.0 - w3 + current
            k3w = eps * (v3 + a - b * w3)
            v4 = v + dt * k3v
            w4 = w + dt * k3w
            k4v = v4 - v4 * v4 * v4 / 3.0 - w4 + current
            k4w = eps * (v4 + a - b * w4)
            v = v + dt * (k1v + 2.0 * k2v + 2.0 * k3v + k4v) / 6.0
            w = w + dt * (k1w + 2.0 * k2w + 2.0 * k3w + k4w) / 6.0
            trace[t] = v
            if v >= thr and v_prev < thr:
                spikes += 1
        return trace, spikes, v, w

    def _simulate_rust(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, float, float]:
        assert _rust_simulate is not None
        trace_list, spikes, vf, wf = _rust_simulate(
            self.v,
            self.w,
            self.a,
            self.b,
            self.epsilon,
            self.dt,
            self.v_threshold,
            n_steps,
            current,
        )
        return np.asarray(trace_list, dtype=np.float64), int(spikes), float(vf), float(wf)

    def _simulate_julia(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, float, float]:
        assert _julia_module is not None
        result = _julia_module.simulate_trace(
            float(self.v),
            float(self.w),
            float(self.a),
            float(self.b),
            float(self.epsilon),
            float(self.dt),
            float(self.v_threshold),
            int(n_steps),
            float(current),
        )
        trace = np.asarray(result.trace, dtype=np.float64)
        return trace, int(result.spikes), float(result.vf), float(result.wf)

    def _simulate_go(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, float, float]:
        assert _go_lib is not None
        import ctypes

        trace = np.zeros(n_steps + 2, dtype=np.float64, order="C")
        spikes = _go_lib.fitzhugh_nagumo_simulate_c(
            ctypes.c_double(self.v),
            ctypes.c_double(self.w),
            ctypes.c_double(self.a),
            ctypes.c_double(self.b),
            ctypes.c_double(self.epsilon),
            ctypes.c_double(self.dt),
            ctypes.c_double(self.v_threshold),
            ctypes.c_int(n_steps),
            ctypes.c_double(current),
            trace.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        vf = float(trace[n_steps]) if n_steps > 0 else self.v
        wf = float(trace[n_steps + 1]) if n_steps > 0 else self.w
        return np.ascontiguousarray(trace[:n_steps]), int(spikes), vf, wf

    def _simulate_mojo(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, float, float]:
        assert _mojo_lib is not None
        trace = np.zeros(n_steps + 2, dtype=np.float64, order="C")
        spikes = _mojo_lib.fitzhugh_nagumo_simulate_c(
            float(self.v),
            float(self.w),
            float(self.a),
            float(self.b),
            float(self.epsilon),
            float(self.dt),
            float(self.v_threshold),
            int(n_steps),
            float(current),
            int(trace.ctypes.data),
        )
        vf = float(trace[n_steps]) if n_steps > 0 else self.v
        wf = float(trace[n_steps + 1]) if n_steps > 0 else self.w
        return np.ascontiguousarray(trace[:n_steps]), int(spikes), vf, wf

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
            # `v * v * v` (not `v**3`) so the cube is exact IEEE multiplication
            # and bit-identical to the Rust `v.powi(3)` / Julia `v^3` / Go/Mojo
            # `v*v*v` used by the polyglot `simulate` backends.
            dv = v - v * v * v / 3.0 - w + current
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
