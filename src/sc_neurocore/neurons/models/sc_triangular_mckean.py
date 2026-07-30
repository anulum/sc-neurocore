# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Retained triangular piecewise-linear oscillator

from __future__ import annotations

import importlib as _importlib
import math
import os as _os
from dataclasses import dataclass
from typing import Callable, ClassVar, Optional

import numpy as np
import numpy.typing as npt

# ───────────────────────── backend detection ─────────────────────────
#
# A single `step` is trivial, but an N-step RK4 simulation is a sequential
# recurrence (each step depends on the previous) that does not vectorise, so a
# compiled inner loop genuinely beats Python. The polyglot chain (Rust PyO3,
# Julia juliacall, Go cgo, Mojo FFI) accelerates `simulate`. The piecewise-linear
# right-hand side is exact arithmetic, so Rust/Julia/Go reproduce the NumPy
# reference bit-for-bit; the FMA-fusing Mojo backend stays within a measured
# per-step ULP band (the two-dimensional autonomous flow cannot be chaotic, so
# that band does not amplify) with identical spike counts.

_RustSimulate = Callable[..., "tuple[list[float], int, float, float]"]


def _load_rust_simulate() -> _RustSimulate:
    engine = _importlib.import_module("sc_neurocore_engine")
    return engine.py_sc_triangular_mckean_simulate  # type: ignore[no-any-return]


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
    jl_path = _os.path.abspath(
        _os.path.join(_ACCEL_ROOT, "julia", "neurons", "sc_triangular_mckean.jl")
    )
    if not _os.path.isfile(jl_path):
        return False
    juliacall = _importlib.import_module("juliacall")
    jl = juliacall.Main
    jl.include(jl_path)
    _julia_module = jl.SCTriangularMcKeanAccel
    _HAS_JULIA = True
    return True


def _ensure_go_loaded() -> bool:
    global _go_lib, _HAS_GO
    if _go_lib is not None:
        return True
    import ctypes

    so_path = _os.path.abspath(
        _os.path.join(_ACCEL_ROOT, "go", "neurons", "mckean", "libsc_triangular_mckean.so")
    )
    if not _os.path.isfile(so_path):
        return False
    try:
        lib = ctypes.CDLL(so_path)
    except OSError:
        return False
    fn = getattr(lib, "sc_triangular_mckean_simulate_c", None)
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

    so_path = _os.path.abspath(
        _os.path.join(_ACCEL_ROOT, "mojo", "neurons", "libsc_triangular_mckean.so")
    )
    if not _os.path.isfile(so_path):
        return False
    try:
        lib = ctypes.CDLL(so_path)
    except OSError:
        return False
    fn = getattr(lib, "sc_triangular_mckean_simulate_c", None)
    if fn is None:
        return False
    fn.argtypes = [ctypes.c_double] * 7 + [ctypes.c_int64, ctypes.c_double, ctypes.c_int64]
    fn.restype = ctypes.c_int64
    _mojo_lib = lib
    _HAS_MOJO = True
    return True


@dataclass
class SCTriangularMcKeanNeuron:
    """Retained SC triangular piecewise-linear FitzHugh-Nagumo oscillator.

    The model evolves the two-state ODE using candidate-first RK4 while
    preserving the three piecewise-linear voltage branches of McKean's
    analytically tractable Nagumo equation.

    This project recurrence is not attributed to McKean's Heaviside system.
    """

    _FINITE_FIELDS: ClassVar[tuple[str, ...]] = ("v", "w", "v_peak")
    _POSITIVE_FIELDS: ClassVar[tuple[str, ...]] = ("epsilon", "gamma", "dt")

    v: float = 0.0
    w: float = 0.0
    a: float = 0.25
    epsilon: float = 0.01
    gamma: float = 0.5
    dt: float = 0.1
    v_peak: float = 0.8

    def __post_init__(self) -> None:
        for name in self._FINITE_FIELDS:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a real finite scalar")
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            setattr(self, name, value)
        if isinstance(self.a, bool) or not isinstance(self.a, (int, float)):
            raise TypeError("a must be a real scalar in the open interval (0, 1)")
        self.a = float(self.a)
        if not math.isfinite(self.a) or not 0.0 < self.a < 1.0:
            raise ValueError("a must be finite and in the open interval (0, 1)")
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
        a = self._finite_float("a", self.a)
        if not 0.0 < a < 1.0:
            raise ValueError("a must be finite and in the open interval (0, 1)")
        for name in self._POSITIVE_FIELDS:
            value = self._finite_float(name, getattr(self, name))
            if value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        return current

    def _f(self, v: float) -> float:
        v = self._finite_float("v", v)
        mid1 = self.a / 2.0
        mid2 = (1.0 + self.a) / 2.0
        if v < mid1:
            return -v
        if v < mid2:
            return v - self.a
        return 1.0 - v

    def _derivatives(self, v: float, w: float, current: float) -> tuple[float, float]:
        if not all(math.isfinite(value) for value in (v, w, current)):
            raise FloatingPointError("SC triangular runtime state and current must be finite")
        dv = self._f(v) - w + current
        dw = self.epsilon * (v - self.gamma * w)
        if not math.isfinite(dv) or not math.isfinite(dw):
            raise FloatingPointError("SC triangular derivative must be finite")
        return dv, dw

    @staticmethod
    def _validate_candidate(v: float, w: float) -> None:
        if not math.isfinite(v) or not math.isfinite(w):
            raise FloatingPointError("SC triangular RK4 candidate must be finite")

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

    def simulate(
        self, n_steps: int, current: float = 0.0, backend: str = "auto"
    ) -> tuple[npt.NDArray[np.float64], int]:
        """Advance ``n_steps`` RK4 updates from the current state, returning ``(trace, spikes)``.

        ``trace[t]`` is the membrane variable ``v`` after step ``t``; ``spikes``
        counts the steps whose ``v`` crossed ``v_peak`` upward. The instance state
        ``(v, w)`` is advanced to the final step. The Rust/Julia/Go backends
        reproduce the pure-NumPy reference bit-for-bit; the FMA-fusing Mojo backend
        stays within a measured per-step ULP band with identical spike counts.
        """
        if n_steps < 0:
            raise ValueError("n_steps must be non-negative")
        if backend not in ("auto", "rust", "julia", "go", "mojo", "python"):
            raise ValueError(f"backend must be auto/rust/julia/go/mojo/python, got {backend!r}")
        current = self._validate_runtime_contract(current)

        if backend == "rust" and not _HAS_RUST:
            raise RuntimeError(
                "Rust SC triangular backend requested but the engine wheel lacks it."
            )
        if backend == "julia" and not _ensure_julia_loaded():
            raise RuntimeError(
                "Julia SC triangular backend requested but juliacall/.jl is unavailable."
            )
        if backend == "go" and not _ensure_go_loaded():
            raise RuntimeError(
                "Go SC triangular backend requested but libsc_triangular_mckean.so is not built; run "
                "`cd src/sc_neurocore/accel/go/neurons/mckean && go build "
                "-buildmode=c-shared -o libsc_triangular_mckean.so mckean.go`."
            )
        if backend == "mojo" and not _ensure_mojo_loaded():
            raise RuntimeError(
                "Mojo SC triangular backend requested but libsc_triangular_mckean.so is not built; run "
                "`cd src/sc_neurocore/accel/mojo/neurons && mojo build --emit shared-lib "
                "-o libsc_triangular_mckean.so mckean.mojo`."
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
        a, eps, gamma, dt, v_peak = self.a, self.epsilon, self.gamma, self.dt, self.v_peak
        half_a = a / 2.0
        mid = (1.0 + a) / 2.0

        def fv(x: float) -> float:
            if x < half_a:
                return -x
            if x < mid:
                return x - a
            return 1.0 - x

        spikes = 0
        for t in range(n_steps):
            v_prev = v
            dv1 = fv(v) - w + current
            dw1 = eps * (v - gamma * w)
            v2 = v + 0.5 * dt * dv1
            w2 = w + 0.5 * dt * dw1
            dv2 = fv(v2) - w2 + current
            dw2 = eps * (v2 - gamma * w2)
            v3 = v + 0.5 * dt * dv2
            w3 = w + 0.5 * dt * dw2
            dv3 = fv(v3) - w3 + current
            dw3 = eps * (v3 - gamma * w3)
            v4 = v + dt * dv3
            w4 = w + dt * dw3
            dv4 = fv(v4) - w4 + current
            dw4 = eps * (v4 - gamma * w4)
            v = v + dt * (dv1 + 2.0 * dv2 + 2.0 * dv3 + dv4) / 6.0
            w = w + dt * (dw1 + 2.0 * dw2 + 2.0 * dw3 + dw4) / 6.0
            trace[t] = v
            if v >= v_peak and v_prev < v_peak:
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
            self.epsilon,
            self.gamma,
            self.dt,
            self.v_peak,
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
            float(self.epsilon),
            float(self.gamma),
            float(self.dt),
            float(self.v_peak),
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
        spikes = _go_lib.sc_triangular_mckean_simulate_c(
            ctypes.c_double(self.v),
            ctypes.c_double(self.w),
            ctypes.c_double(self.a),
            ctypes.c_double(self.epsilon),
            ctypes.c_double(self.gamma),
            ctypes.c_double(self.dt),
            ctypes.c_double(self.v_peak),
            ctypes.c_int(n_steps),
            ctypes.c_double(current),
            trace.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        vf = float(trace[n_steps]) if n_steps > 0 else self.v
        wf = float(trace[n_steps + 1])
        return np.ascontiguousarray(trace[:n_steps]), int(spikes), vf, wf

    def _simulate_mojo(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, float, float]:
        assert _mojo_lib is not None
        trace = np.zeros(n_steps + 2, dtype=np.float64, order="C")
        spikes = _mojo_lib.sc_triangular_mckean_simulate_c(
            float(self.v),
            float(self.w),
            float(self.a),
            float(self.epsilon),
            float(self.gamma),
            float(self.dt),
            float(self.v_peak),
            int(n_steps),
            float(current),
            int(trace.ctypes.data),
        )
        vf = float(trace[n_steps]) if n_steps > 0 else self.v
        wf = float(trace[n_steps + 1])
        return np.ascontiguousarray(trace[:n_steps]), int(spikes), vf, wf

    def reset(self) -> None:
        self.v = 0.0
        self.w = 0.0
