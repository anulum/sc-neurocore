# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — FitzHugh 1976 / Rinzel 1987 — FHN + slow variable

from __future__ import annotations

import importlib as _importlib
import math
import os as _os
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import numpy.typing as npt

_STATE_NAMES = ("v", "w", "y")
_PARAMETER_NAMES = ("a", "b", "c", "d", "delta", "mu", "dt", "v_threshold")
_POSITIVE_PARAMETERS = ("b", "d", "delta", "mu", "dt")

# ───────────────────────── backend detection ─────────────────────────
#
# `step` runs one RK4 update; an N-step `simulate` is a sequential recurrence
# (each step depends on the previous) that does not vectorise, so a compiled
# inner loop genuinely beats Python. The Hindmarsh-Rose-style right-hand side is
# exact arithmetic (a cube written `v*v*v`, additions and multiplications — no
# transcendental functions), so Rust, Julia and Go reproduce the NumPy trace
# bit-for-bit; Mojo agrees to a small, non-amplifying ULP bound.

_RustSimulate = Callable[..., "tuple[list[float], int, float, float, float]"]


def _load_rust_simulate() -> _RustSimulate:
    engine = _importlib.import_module("sc_neurocore_engine")
    return engine.py_fitzhugh_rinzel_simulate  # type: ignore[no-any-return]


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
    jl_path = _os.path.abspath(_os.path.join(_ACCEL_ROOT, "julia", "neurons", "fitzhugh_rinzel.jl"))
    if not _os.path.isfile(jl_path):
        return False
    juliacall = _importlib.import_module("juliacall")
    jl = juliacall.Main
    jl.include(jl_path)
    _julia_module = jl.FitzHughRinzelAccel
    _HAS_JULIA = True
    return True


def _ensure_go_loaded() -> bool:
    global _go_lib, _HAS_GO
    if _go_lib is not None:
        return True
    import ctypes

    so_path = _os.path.abspath(
        _os.path.join(_ACCEL_ROOT, "go", "neurons", "fitzhugh_rinzel", "libfhr.so")
    )
    if not _os.path.isfile(so_path):
        return False
    try:
        lib = ctypes.CDLL(so_path)
    except OSError:
        return False
    fn = getattr(lib, "fitzhugh_rinzel_simulate_c", None)
    if fn is None:
        return False
    fn.argtypes = [ctypes.c_double] * 11 + [
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

    so_path = _os.path.abspath(_os.path.join(_ACCEL_ROOT, "mojo", "neurons", "libfhr.so"))
    if not _os.path.isfile(so_path):
        return False
    try:
        lib = ctypes.CDLL(so_path)
    except OSError:
        return False
    fn = getattr(lib, "fitzhugh_rinzel_simulate_c", None)
    if fn is None:
        return False
    # 11 float params + n_steps + current + trace addr; returns spikes.
    fn.argtypes = [ctypes.c_double] * 11 + [ctypes.c_int64, ctypes.c_double, ctypes.c_int64]
    fn.restype = ctypes.c_int64
    _mojo_lib = lib
    _HAS_MOJO = True
    return True


@dataclass
class FitzHughRinzelNeuron:
    """FitzHugh-Rinzel three-state qualitative bursting model.

    dv/dt = v - v^3/3 - w + y + I
    dw/dt = delta * (a + v - b*w)
    dy/dt = mu * (c - v - d*y)

    Runtime integration uses RK4 over the published three-state ODE with
    current held constant for one step.
    """

    v: float = -1.0
    w: float = -0.5
    y: float = 0.0
    a: float = 0.7
    b: float = 0.8
    c: float = -0.775
    d: float = 1.0
    delta: float = 0.08
    mu: float = 0.0001
    dt: float = 0.1
    v_threshold: float = 1.0

    def __post_init__(self) -> None:
        self._validate_numeric_contract()

    @staticmethod
    def _finite_float(name: str, value: float) -> float:
        if isinstance(value, bool):
            raise ValueError(f"FitzHugh-Rinzel parameter {name} must be finite")
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"FitzHugh-Rinzel parameter {name} must be finite") from exc
        if not math.isfinite(result):
            raise ValueError(f"FitzHugh-Rinzel parameter {name} must be finite")
        return result

    def _numeric_fields(self) -> tuple[tuple[str, float], ...]:
        return tuple((name, getattr(self, name)) for name in (*_STATE_NAMES, *_PARAMETER_NAMES))

    def _validate_numeric_contract(self) -> None:
        validated = {
            name: self._finite_float(name, value) for name, value in self._numeric_fields()
        }
        for name in _POSITIVE_PARAMETERS:
            if validated[name] <= 0.0:
                raise ValueError(f"FitzHugh-Rinzel parameter {name} must be positive")
        for name, value in validated.items():
            setattr(self, name, value)

    def _derivatives(
        self, v: float, w: float, y: float, current: float
    ) -> tuple[float, float, float]:
        if not all(math.isfinite(value) for value in (v, w, y, current)):
            raise FloatingPointError("FitzHugh-Rinzel runtime state and current must be finite")
        # `v * v * v` (not `v**3`) so the cube is exact IEEE multiplication and
        # bit-identical to the Rust `v.powi(3)` / Julia `v^3` / Go/Mojo `v*v*v`
        # used by the polyglot `simulate` backends. Exact multiplication overflows
        # to +/-inf (it does not raise OverflowError as `v**3` did), so the finite
        # guard below is what rejects an exploding derivative.
        dv = v - v * v * v / 3.0 - w + y + current
        dw = self.delta * (self.a + v - self.b * w)
        dy = self.mu * (self.c - v - self.d * y)
        if not all(math.isfinite(value) for value in (dv, dw, dy)):
            raise FloatingPointError("FitzHugh-Rinzel derivative must be finite")
        return dv, dw, dy

    def _rk4_candidate(self, current: float) -> tuple[float, float, float]:
        v0, w0, y0, dt = self.v, self.w, self.y, self.dt
        k1 = self._derivatives(v0, w0, y0, current)
        k2 = self._derivatives(
            v0 + 0.5 * dt * k1[0],
            w0 + 0.5 * dt * k1[1],
            y0 + 0.5 * dt * k1[2],
            current,
        )
        k3 = self._derivatives(
            v0 + 0.5 * dt * k2[0],
            w0 + 0.5 * dt * k2[1],
            y0 + 0.5 * dt * k2[2],
            current,
        )
        k4 = self._derivatives(
            v0 + dt * k3[0],
            w0 + dt * k3[1],
            y0 + dt * k3[2],
            current,
        )
        return (
            v0 + dt * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0,
            w0 + dt * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0,
            y0 + dt * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]) / 6.0,
        )

    @staticmethod
    def _validate_candidate(v: float, w: float, y: float) -> tuple[float, float, float]:
        if not all(math.isfinite(value) for value in (v, w, y)):
            raise FloatingPointError("FitzHugh-Rinzel candidate state must be finite")
        return float(v), float(w), float(y)

    def step(self, current: float) -> int:
        """Advance the model by one RK4 step."""

        self._validate_numeric_contract()
        current = self._finite_float("current", current)
        v_prev = self.v
        self.v, self.w, self.y = self._validate_candidate(*self._rk4_candidate(current))
        return 1 if (self.v >= self.v_threshold and v_prev < self.v_threshold) else 0

    def simulate(
        self, n_steps: int, current: float = 0.0, backend: str = "auto"
    ) -> tuple[npt.NDArray[np.float64], int]:
        """Advance ``n_steps`` RK4 steps from the current state, returning ``(trace, spikes)``.

        ``trace[t]`` is the membrane variable ``v`` after step ``t``; ``spikes``
        counts upward crossings of ``v_threshold``. The instance state
        ``(v, w, y)`` is advanced to the final step. The right-hand side is exact
        arithmetic, so Rust, Julia and Go reproduce the pure-NumPy reference
        bit-for-bit; Mojo agrees to a documented, non-amplifying ULP bound.
        A rejected derivative, candidate, or native result leaves the complete
        pre-batch state unchanged.
        """
        if n_steps < 0:
            raise ValueError("n_steps must be non-negative")
        current = self._finite_float("current", current)
        if backend not in ("auto", "rust", "julia", "go", "mojo", "python"):
            raise ValueError(f"backend must be auto/rust/julia/go/mojo/python, got {backend!r}")
        self._validate_numeric_contract()

        if backend == "rust" and not _HAS_RUST:
            raise RuntimeError("Rust FitzHugh-Rinzel backend requested but the engine lacks it.")
        if backend == "julia" and not _ensure_julia_loaded():
            raise RuntimeError("Julia FitzHugh-Rinzel backend requested but it is unavailable.")
        if backend == "go" and not _ensure_go_loaded():
            raise RuntimeError(
                "Go FitzHugh-Rinzel backend requested but libfhr.so is not built; run "
                "`cd src/sc_neurocore/accel/go/neurons/fitzhugh_rinzel && go build "
                "-buildmode=c-shared -o libfhr.so fitzhugh_rinzel.go`."
            )
        if backend == "mojo" and not _ensure_mojo_loaded():
            raise RuntimeError(
                "Mojo FitzHugh-Rinzel backend requested but libfhr.so is not built; run "
                "`cd src/sc_neurocore/accel/mojo/neurons && mojo build --emit shared-lib "
                "-o libfhr.so fitzhugh_rinzel.mojo`."
            )

        if backend == "rust" or (backend == "auto" and _HAS_RUST):
            trace, spikes, vf, wf, yf = self._simulate_rust(n_steps, current)
        elif backend == "julia":
            trace, spikes, vf, wf, yf = self._simulate_julia(n_steps, current)
        elif backend == "go":
            trace, spikes, vf, wf, yf = self._simulate_go(n_steps, current)
        elif backend == "mojo":
            trace, spikes, vf, wf, yf = self._simulate_mojo(n_steps, current)
        else:
            trace, spikes, vf, wf, yf = self._simulate_python(n_steps, current)
        trace, spikes, vf, wf, yf = self._validate_batch_result(
            backend=backend,
            n_steps=n_steps,
            trace=trace,
            spikes=spikes,
            final_state=(vf, wf, yf),
        )
        self.v, self.w, self.y = vf, wf, yf
        return trace, spikes

    @staticmethod
    def _validate_batch_result(
        *,
        backend: str,
        n_steps: int,
        trace: npt.NDArray[np.float64],
        spikes: int,
        final_state: tuple[float, float, float],
    ) -> tuple[npt.NDArray[np.float64], int, float, float, float]:
        """Validate a complete native result before the caller commits state."""
        values = np.asarray(trace, dtype=np.float64)
        if (
            spikes < 0
            or values.shape != (n_steps,)
            or not np.all(np.isfinite(values))
            or not all(math.isfinite(value) for value in final_state)
        ):
            raise FloatingPointError(
                f"FitzHugh-Rinzel {backend} batch produced an invalid candidate"
            )
        return values, int(spikes), final_state[0], final_state[1], final_state[2]

    def _simulate_python(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, float, float, float]:
        trace = np.empty(n_steps, dtype=np.float64)
        v, w, y = self.v, self.w, self.y
        a, b, c, d = self.a, self.b, self.c, self.d
        delta, mu, dt, thr = self.delta, self.mu, self.dt, self.v_threshold
        spikes = 0

        def deriv(vv: float, ww: float, yy: float) -> tuple[float, float, float]:
            try:
                candidate = (
                    vv - vv * vv * vv / 3.0 - ww + yy + current,
                    delta * (a + vv - b * ww),
                    mu * (c - vv - d * yy),
                )
            except OverflowError as exc:
                raise FloatingPointError(
                    "FitzHugh-Rinzel Python batch derivative overflowed"
                ) from exc
            if not all(math.isfinite(value) for value in candidate):
                raise FloatingPointError(
                    "FitzHugh-Rinzel Python batch derivative became non-finite"
                )
            return candidate

        for t in range(n_steps):
            v_prev = v
            k1v, k1w, k1y = deriv(v, w, y)
            k2v, k2w, k2y = deriv(v + 0.5 * dt * k1v, w + 0.5 * dt * k1w, y + 0.5 * dt * k1y)
            k3v, k3w, k3y = deriv(v + 0.5 * dt * k2v, w + 0.5 * dt * k2w, y + 0.5 * dt * k2y)
            k4v, k4w, k4y = deriv(v + dt * k3v, w + dt * k3w, y + dt * k3y)
            candidate = (
                v + dt * (k1v + 2.0 * k2v + 2.0 * k3v + k4v) / 6.0,
                w + dt * (k1w + 2.0 * k2w + 2.0 * k3w + k4w) / 6.0,
                y + dt * (k1y + 2.0 * k2y + 2.0 * k3y + k4y) / 6.0,
            )
            if not all(math.isfinite(value) for value in candidate):
                raise FloatingPointError("FitzHugh-Rinzel Python batch candidate became non-finite")
            v, w, y = candidate
            trace[t] = v
            if v >= thr and v_prev < thr:
                spikes += 1
        return trace, spikes, v, w, y

    def _simulate_rust(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, float, float, float]:
        assert _rust_simulate is not None
        trace_list, spikes, vf, wf, yf = _rust_simulate(
            self.v,
            self.w,
            self.y,
            self.a,
            self.b,
            self.c,
            self.d,
            self.delta,
            self.mu,
            self.dt,
            self.v_threshold,
            n_steps,
            current,
        )
        return (
            np.asarray(trace_list, dtype=np.float64),
            int(spikes),
            float(vf),
            float(wf),
            float(yf),
        )

    def _simulate_julia(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, float, float, float]:
        assert _julia_module is not None
        try:
            result = _julia_module.simulate_trace(
                float(self.v),
                float(self.w),
                float(self.y),
                float(self.a),
                float(self.b),
                float(self.c),
                float(self.d),
                float(self.delta),
                float(self.mu),
                float(self.dt),
                float(self.v_threshold),
                int(n_steps),
                float(current),
            )
        except Exception as exc:
            raise FloatingPointError(
                "FitzHugh-Rinzel Julia batch rejected an invalid candidate"
            ) from exc
        trace = np.asarray(result.trace, dtype=np.float64)
        return trace, int(result.spikes), float(result.vf), float(result.wf), float(result.yf)

    def _simulate_go(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, float, float, float]:
        assert _go_lib is not None
        import ctypes

        trace = np.zeros(n_steps + 3, dtype=np.float64, order="C")
        spikes = _go_lib.fitzhugh_rinzel_simulate_c(
            ctypes.c_double(self.v),
            ctypes.c_double(self.w),
            ctypes.c_double(self.y),
            ctypes.c_double(self.a),
            ctypes.c_double(self.b),
            ctypes.c_double(self.c),
            ctypes.c_double(self.d),
            ctypes.c_double(self.delta),
            ctypes.c_double(self.mu),
            ctypes.c_double(self.dt),
            ctypes.c_double(self.v_threshold),
            ctypes.c_int(n_steps),
            ctypes.c_double(current),
            trace.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        vf = float(trace[n_steps]) if n_steps > 0 else self.v
        wf = float(trace[n_steps + 1]) if n_steps > 0 else self.w
        yf = float(trace[n_steps + 2]) if n_steps > 0 else self.y
        return np.ascontiguousarray(trace[:n_steps]), int(spikes), vf, wf, yf

    def _simulate_mojo(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, float, float, float]:
        assert _mojo_lib is not None
        trace = np.zeros(n_steps + 3, dtype=np.float64, order="C")
        spikes = _mojo_lib.fitzhugh_rinzel_simulate_c(
            float(self.v),
            float(self.w),
            float(self.y),
            float(self.a),
            float(self.b),
            float(self.c),
            float(self.d),
            float(self.delta),
            float(self.mu),
            float(self.dt),
            float(self.v_threshold),
            int(n_steps),
            float(current),
            int(trace.ctypes.data),
        )
        vf = float(trace[n_steps]) if n_steps > 0 else self.v
        wf = float(trace[n_steps + 1]) if n_steps > 0 else self.w
        yf = float(trace[n_steps + 2]) if n_steps > 0 else self.y
        return np.ascontiguousarray(trace[:n_steps]), int(spikes), vf, wf, yf

    def reset(self) -> None:
        self.v, self.w, self.y = -1.0, -0.5, 0.0
