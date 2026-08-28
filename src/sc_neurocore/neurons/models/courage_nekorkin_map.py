# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Courbage, Nekorkin & Vdovin 2007 discontinuous map

"""Source-faithful Courbage-Nekorkin-Vdovin spiking-bursting map.

The autonomous recurrence is equations 3–5 of Courbage, Nekorkin, and Vdovin
(2007):

    x[n+1] = x[n] + F(x[n]) - y[n] - beta*H(x[n] - d)
    y[n+1] = y[n] + eps*(x[n] - J)

F has the paper's three linear branches and H(0)=1. Defaults reproduce the
complete Figure-4 chaotic spiking-bursting parameter profile. The initial state
(0, 0) and upward crossing of x_threshold=d are disclosed SC-NeuroCore protocol
choices. current is an additive perturbation of the fast recurrence, so zero
current recovers the published autonomous map.

Reference: Courbage, M., Nekorkin, V. I. & Vdovin, L. V. (2007), Chaotic
oscillations in a map-based model of neural activity, Chaos 17, 043109.
DOI: 10.1063/1.2795435; arXiv:0712.2097.
"""

from __future__ import annotations

import ctypes
import importlib
import importlib.util
import math
import os
from dataclasses import dataclass
from typing import Any, Protocol, cast

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True)
class _CourbageParameters:
    m0: float
    m1: float
    a: float
    d: float
    j: float
    beta: float
    eps: float
    x_threshold: float

    def float_args(self, x: float, y: float) -> tuple[float, ...]:
        """Return the stable cross-language floating-point ABI order."""
        return (
            x,
            y,
            self.m0,
            self.m1,
            self.a,
            self.d,
            self.j,
            self.beta,
            self.eps,
            self.x_threshold,
        )

    def breakpoints(self) -> tuple[float, float]:
        """Return the source branch boundaries (Jmin, Jmax)."""
        am1 = self.a * self.m1
        denominator = self.m0 + self.m1
        return am1 / denominator, (self.m0 + am1) / denominator

    def validate(self, x: float, y: float) -> None:
        """Reject states or parameters outside the paper's analysed domain."""
        if not all(math.isfinite(value) for value in self.float_args(x, y)):
            raise ValueError("Courbage state and parameters must be finite")
        if not 0.0 < self.m0 < 1.0:
            raise ValueError("m0 must satisfy 0 < m0 < 1")
        if self.m1 <= 0.0:
            raise ValueError("m1 must be positive")
        if not 0.0 < self.a < 1.0:
            raise ValueError("a must satisfy 0 < a < 1")
        if self.d <= 0.0 or self.beta <= 0.0 or self.eps <= 0.0:
            raise ValueError("d, beta, and eps must be positive")
        j_min, j_max = self.breakpoints()
        if not 0.0 < self.j < self.d:
            raise ValueError("J must satisfy 0 < J < d")
        if not j_min < self.d < j_max:
            raise ValueError("d must satisfy Jmin < d < Jmax")

    def candidate(self, x: float, y: float, current: float) -> tuple[float, float]:
        """Evaluate one simultaneous source-map update."""
        j_min, j_max = self.breakpoints()
        if x <= j_min:
            field = -self.m0 * x
        elif x < j_max:
            field = self.m1 * (x - self.a)
        else:
            field = -self.m0 * (x - 1.0)
        heaviside = 1.0 if x >= self.d else 0.0
        x_new = x + field - y - self.beta * heaviside + current
        y_new = y + self.eps * (x - self.j)
        if not math.isfinite(x_new) or not math.isfinite(y_new):
            raise FloatingPointError("Courbage map candidate became non-finite")
        return x_new, y_new


class _RustSimulate(Protocol):
    def __call__(self, *args: Any) -> tuple[list[float], int, float, float]: ...


class _JuliaResult(Protocol):
    trace: Any
    spikes: int
    xf: float
    yf: float


class _JuliaAccel(Protocol):
    def simulate_trace(self, *args: Any) -> _JuliaResult: ...


def _load_rust_simulate() -> _RustSimulate:
    engine = importlib.import_module("sc_neurocore_engine")
    return cast(_RustSimulate, engine.py_courage_nekorkin_map_simulate)


try:
    _rust_simulate: _RustSimulate | None = _load_rust_simulate()
    _HAS_RUST = True
except (ImportError, AttributeError):
    _rust_simulate = None
    _HAS_RUST = False

_julia_module: _JuliaAccel | None = None
_go_lib: ctypes.CDLL | None = None
_mojo_lib: ctypes.CDLL | None = None
_ACCEL_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "accel")
_MAX_C_STEPS = (1 << 31) - 1


def _ensure_julia_loaded() -> bool:
    global _julia_module
    if _julia_module is not None:
        return True
    if importlib.util.find_spec("juliacall") is None:
        return False
    path = os.path.abspath(os.path.join(_ACCEL_ROOT, "julia", "neurons", "courage_nekorkin_map.jl"))
    if not os.path.isfile(path):
        return False
    julia = importlib.import_module("juliacall").Main
    julia.include(path)
    _julia_module = cast(_JuliaAccel, julia.CourageNekorkinMapAccel)
    return True


def _load_c_backend(path: str, *, mojo: bool) -> ctypes.CDLL | None:
    if not os.path.isfile(path):
        return None
    try:
        library = ctypes.CDLL(path)
    except OSError:
        return None
    function = getattr(library, "courage_nekorkin_map_simulate_c", None)
    if function is None:
        return None
    if mojo:
        function.argtypes = [ctypes.c_double] * 10 + [
            ctypes.c_int64,
            ctypes.c_double,
            ctypes.c_int64,
        ]
    else:
        function.argtypes = [ctypes.c_double] * 10 + [
            ctypes.c_int,
            ctypes.c_double,
            ctypes.POINTER(ctypes.c_double),
        ]
    function.restype = ctypes.c_longlong
    return library


def _ensure_go_loaded() -> bool:
    global _go_lib
    if _go_lib is not None:
        return True
    path = os.path.abspath(
        os.path.join(
            _ACCEL_ROOT,
            "go",
            "neurons",
            "courage_nekorkin_map",
            "libcourage.so",
        )
    )
    _go_lib = _load_c_backend(path, mojo=False)
    return _go_lib is not None


def _ensure_mojo_loaded() -> bool:
    global _mojo_lib
    if _mojo_lib is not None:
        return True
    path = os.path.abspath(os.path.join(_ACCEL_ROOT, "mojo", "neurons", "libcourage.so"))
    _mojo_lib = _load_c_backend(path, mojo=True)
    return _mojo_lib is not None


@dataclass
class CourageNekorkinMapNeuron:
    """Courbage-Nekorkin-Vdovin map with the source Figure-4 profile."""

    x: float = 0.0
    y: float = 0.0
    m0: float = 0.4
    m1: float = 0.65
    a: float = 0.2
    d: float = 0.3
    j: float = 0.13
    beta: float = 0.25
    eps: float = 0.002
    x_threshold: float = 0.3

    def __post_init__(self) -> None:
        self._parameters().validate(self.x, self.y)

    def _parameters(self) -> _CourbageParameters:
        return _CourbageParameters(
            self.m0,
            self.m1,
            self.a,
            self.d,
            self.j,
            self.beta,
            self.eps,
            self.x_threshold,
        )

    def _breakpoints(self) -> tuple[float, float]:
        """Return the source branch boundaries (Jmin, Jmax)."""
        return self._parameters().breakpoints()

    def _f(self, x: float) -> float:
        """Evaluate the paper's piecewise-linear F(x)."""
        parameters = self._parameters()
        j_min, j_max = parameters.breakpoints()
        if x <= j_min:
            return -parameters.m0 * x
        if x < j_max:
            return parameters.m1 * (x - parameters.a)
        return -parameters.m0 * (x - 1.0)

    def step(self, current: float = 0.0) -> int:
        """Advance one map iteration; rejected candidates leave state unchanged."""
        if not math.isfinite(current):
            raise ValueError("current must be finite")
        parameters = self._parameters()
        parameters.validate(self.x, self.y)
        x_new, y_new = parameters.candidate(self.x, self.y, current)
        event = int(x_new >= self.x_threshold and self.x < self.x_threshold)
        self.x, self.y = x_new, y_new
        return event

    def simulate(
        self, n_steps: int, current: float = 0.0, backend: str = "auto"
    ) -> tuple[npt.NDArray[np.float64], int]:
        """Return the post-step x trace and upward-crossing event count."""
        if isinstance(n_steps, bool) or not isinstance(n_steps, int):
            raise ValueError("n_steps must be an integer")
        if not 0 <= n_steps <= _MAX_C_STEPS:
            raise ValueError(f"n_steps must be between 0 and {_MAX_C_STEPS}")
        if not math.isfinite(current):
            raise ValueError("current must be finite")
        if backend not in ("auto", "rust", "julia", "go", "mojo", "python"):
            raise ValueError(f"backend must be auto/rust/julia/go/mojo/python, got {backend!r}")
        self._parameters().validate(self.x, self.y)

        selected = "rust" if backend == "auto" and _HAS_RUST else backend
        if selected == "auto":
            selected = "python"
        if selected == "rust":
            if not _HAS_RUST:
                raise RuntimeError("Rust Courbage backend is unavailable")
            trace, events, x_final, y_final = self._simulate_rust(n_steps, current)
        elif selected == "julia":
            if not _ensure_julia_loaded():
                raise RuntimeError("Julia Courbage backend is unavailable")
            trace, events, x_final, y_final = self._simulate_julia(n_steps, current)
        elif selected == "go":
            if not _ensure_go_loaded():
                raise RuntimeError("Go Courbage backend is unavailable")
            trace, events, x_final, y_final = self._simulate_go(n_steps, current)
        elif selected == "mojo":
            if not _ensure_mojo_loaded():
                raise RuntimeError("Mojo Courbage backend is unavailable")
            trace, events, x_final, y_final = self._simulate_mojo(n_steps, current)
        else:
            trace, events, x_final, y_final = self._simulate_python(n_steps, current)
        self.x, self.y = x_final, y_final
        return trace, events

    def _simulate_python(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, float, float]:
        parameters = self._parameters()
        x, y = self.x, self.y
        trace = np.empty(n_steps, dtype=np.float64)
        events = 0
        for index in range(n_steps):
            x_new, y_new = parameters.candidate(x, y, current)
            events += int(x_new >= parameters.x_threshold and x < parameters.x_threshold)
            x, y = x_new, y_new
            trace[index] = x
        return trace, events, x, y

    def _runtime_args(self) -> tuple[float, ...]:
        return self._parameters().float_args(self.x, self.y)

    def _simulate_rust(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, float, float]:
        assert _rust_simulate is not None
        trace, events, x_final, y_final = _rust_simulate(*self._runtime_args(), n_steps, current)
        return (
            np.asarray(trace, dtype=np.float64),
            int(events),
            float(x_final),
            float(y_final),
        )

    def _simulate_julia(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, float, float]:
        assert _julia_module is not None
        result = _julia_module.simulate_trace(*self._runtime_args(), n_steps, current)
        return (
            np.asarray(result.trace, dtype=np.float64),
            int(result.spikes),
            float(result.xf),
            float(result.yf),
        )

    def _simulate_go(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, float, float]:
        assert _go_lib is not None
        trace = np.zeros(n_steps + 2, dtype=np.float64, order="C")
        events = _go_lib.courage_nekorkin_map_simulate_c(
            *(ctypes.c_double(value) for value in self._runtime_args()),
            ctypes.c_int(n_steps),
            ctypes.c_double(current),
            trace.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        if events < 0:
            raise FloatingPointError("Go Courbage backend rejected the trajectory")
        return (
            np.ascontiguousarray(trace[:n_steps]),
            int(events),
            float(trace[-2]),
            float(trace[-1]),
        )

    def _simulate_mojo(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, float, float]:
        assert _mojo_lib is not None
        trace = np.zeros(n_steps + 2, dtype=np.float64, order="C")
        events = _mojo_lib.courage_nekorkin_map_simulate_c(
            *self._runtime_args(), n_steps, current, int(trace.ctypes.data)
        )
        if events < 0:
            raise FloatingPointError("Mojo Courbage backend rejected the trajectory")
        return (
            np.ascontiguousarray(trace[:n_steps]),
            int(events),
            float(trace[-2]),
            float(trace[-1]),
        )

    def reset(self) -> None:
        """Restore the source-profile protocol initial state."""
        self.x = 0.0
        self.y = 0.0
