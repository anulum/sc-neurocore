# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Cazelles-Courbage-Rabinovich 2001 piecewise-linear map

"""Source-faithful Cazelles-Courbage-Rabinovich bursting map.

The paper defines ``x[n+1] = f(x[n]) + alpha*x[n]**m`` with four
piecewise-linear branches for ``f``. Defaults reproduce Figure 1. The source
writes strict inequalities and leaves exact breakpoints undefined; this
implementation discloses a deterministic right-continuous convention:
``[x0,x1)``, ``[x1,x2)``, ``[x2,x3)``, and ``[x3,x4]``. ``current`` is an
additive maintained perturbation, so zero current recovers the published map.

One catalogue event marks entry into the slow regime: ``x_pre >= x1`` and
``x_post < x1``. This operationalises the paper's slow-regime minima as burst
cycle markers; it is not a paper-defined action-potential threshold.

Reference: Cazelles, B., Courbage, M. & Rabinovich, M. (2001), *Anti-phase
regularization of coupled chaotic maps modelling bursting neurons*,
Europhysics Letters 56(4), 504-509. DOI: 10.1209/epl/i2001-00548-y.
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
class _CazellesParameters:
    alpha: float
    exponent: int
    x0: float
    x1: float
    x2: float
    x3: float
    x4: float
    a1: float
    a2: float
    a3: float
    a4: float
    b1: float
    b2: float
    b3: float
    b4: float

    def float_args(self, x: float) -> tuple[float, ...]:
        """Return the stable cross-language floating-point ABI order."""
        return (
            x,
            self.alpha,
            self.x0,
            self.x1,
            self.x2,
            self.x3,
            self.x4,
            self.a1,
            self.a2,
            self.a3,
            self.a4,
            self.b1,
            self.b2,
            self.b3,
            self.b4,
        )

    def validate(self, x: float) -> None:
        if not all(math.isfinite(value) for value in self.float_args(x)):
            raise ValueError("Cazelles state and parameters must be finite")
        if not 0.0 <= self.alpha < 1.0:
            raise ValueError("alpha must satisfy 0 <= alpha < 1")
        if isinstance(self.exponent, bool) or self.exponent not in (1, 2):
            raise ValueError("exponent must be integer 1 or 2")
        if not self.x0 < self.x1 < self.x2 < self.x3 < self.x4:
            raise ValueError("Cazelles branch bounds must be strictly increasing")
        if not self.x0 <= x <= self.x4:
            raise ValueError("x must lie in the configured Cazelles map domain")

    def candidate(self, x: float, current: float) -> float:
        """Evaluate the map with the disclosed breakpoint convention."""
        if x < self.x1:
            base = self.a1 + self.b1 * x
        elif x < self.x2:
            base = self.a2 + self.b2 * x
        elif x < self.x3:
            base = self.a3 + self.b3 * x
        else:
            base = self.a4 + self.b4 * x
        power = x if self.exponent == 1 else x * x
        candidate = base + self.alpha * power + current
        if not math.isfinite(candidate):
            raise FloatingPointError("Cazelles map candidate became non-finite")
        tolerance = 8.0 * math.ulp(max(1.0, abs(self.x0), abs(self.x4)))
        if self.x0 - tolerance <= candidate < self.x0:
            candidate = self.x0
        elif self.x4 < candidate <= self.x4 + tolerance:
            candidate = self.x4
        if not self.x0 <= candidate <= self.x4:
            raise FloatingPointError("Cazelles map candidate left its configured domain")
        return candidate


class _RustSimulate(Protocol):
    def __call__(self, *args: Any) -> tuple[list[float], int, float]: ...


class _JuliaResult(Protocol):
    trace: Any
    events: int
    xf: float


class _JuliaAccel(Protocol):
    def simulate_trace(self, *args: Any) -> _JuliaResult: ...


def _load_rust_simulate() -> _RustSimulate:
    engine = importlib.import_module("sc_neurocore_engine")
    return cast(_RustSimulate, engine.py_cazelles_map_simulate)


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
_FLOAT_ARGUMENTS = 15
_MAX_C_STEPS = (1 << 31) - 1


def _ensure_julia_loaded() -> bool:
    global _julia_module
    if _julia_module is not None:
        return True
    if importlib.util.find_spec("juliacall") is None:
        return False
    path = os.path.abspath(os.path.join(_ACCEL_ROOT, "julia", "neurons", "cazelles_map.jl"))
    if not os.path.isfile(path):
        return False
    julia = importlib.import_module("juliacall").Main
    julia.include(path)
    _julia_module = cast(_JuliaAccel, julia.CazellesMapAccel)
    return True


def _load_c_backend(path: str, symbol: str, *, mojo: bool) -> ctypes.CDLL | None:
    if not os.path.isfile(path):
        return None
    try:
        library = ctypes.CDLL(path)
    except OSError:
        return None
    function = getattr(library, symbol, None)
    if function is None:
        return None
    if mojo:
        function.argtypes = [ctypes.c_double] * _FLOAT_ARGUMENTS + [
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_double,
            ctypes.c_int64,
        ]
    else:
        function.argtypes = [ctypes.c_double] * _FLOAT_ARGUMENTS + [
            ctypes.c_int,
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
        os.path.join(_ACCEL_ROOT, "go", "neurons", "cazelles_map", "libcazelles.so")
    )
    _go_lib = _load_c_backend(path, "cazelles_map_simulate_c", mojo=False)
    return _go_lib is not None


def _ensure_mojo_loaded() -> bool:
    global _mojo_lib
    if _mojo_lib is not None:
        return True
    path = os.path.abspath(os.path.join(_ACCEL_ROOT, "mojo", "neurons", "libcazelles.so"))
    _mojo_lib = _load_c_backend(path, "cazelles_map_simulate_c", mojo=True)
    return _mojo_lib is not None


@dataclass
class CazellesMapNeuron:
    """Cazelles et al. (2001) scalar four-branch bursting map."""

    x: float = 0.1
    alpha: float = 0.0
    exponent: int = 2
    x0: float = 0.0
    x1: float = 0.4
    x2: float = 0.6
    x3: float = 0.7
    x4: float = 1.0
    a1: float = 0.0
    a2: float = 1.5
    a3: float = -0.9
    a4: float = 1.4
    b1: float = 1.05
    b2: float = -1.25
    b3: float = 1.5
    b4: float = -1.0

    def __post_init__(self) -> None:
        self._parameters().validate(self.x)

    def _parameters(self) -> _CazellesParameters:
        return _CazellesParameters(
            self.alpha,
            self.exponent,
            self.x0,
            self.x1,
            self.x2,
            self.x3,
            self.x4,
            self.a1,
            self.a2,
            self.a3,
            self.a4,
            self.b1,
            self.b2,
            self.b3,
            self.b4,
        )

    def step(self, current: float) -> int:
        """Advance once; rejected candidates leave ``x`` unchanged."""
        if not math.isfinite(current):
            raise ValueError("current must be finite")
        parameters = self._parameters()
        parameters.validate(self.x)
        candidate = parameters.candidate(self.x, current)
        event = int(self.x >= self.x1 and candidate < self.x1)
        self.x = candidate
        return event

    def simulate(
        self, n_steps: int, current: float = 0.0, backend: str = "auto"
    ) -> tuple[npt.NDArray[np.float64], int]:
        """Return the post-step scalar trace and slow-regime entry count."""
        if isinstance(n_steps, bool) or not isinstance(n_steps, int):
            raise ValueError("n_steps must be an integer")
        if not 0 <= n_steps <= _MAX_C_STEPS:
            raise ValueError(f"n_steps must be between 0 and {_MAX_C_STEPS}")
        if not math.isfinite(current):
            raise ValueError("current must be finite")
        if backend not in ("auto", "rust", "julia", "go", "mojo", "python"):
            raise ValueError(f"backend must be auto/rust/julia/go/mojo/python, got {backend!r}")
        self._parameters().validate(self.x)

        selected = "rust" if backend == "auto" and _HAS_RUST else backend
        if selected == "auto":
            selected = "python"
        if selected == "rust":
            if not _HAS_RUST:
                raise RuntimeError("Rust Cazelles backend is unavailable")
            trace, events, xf = self._simulate_rust(n_steps, current)
        elif selected == "julia":
            if not _ensure_julia_loaded():
                raise RuntimeError("Julia Cazelles backend is unavailable")
            trace, events, xf = self._simulate_julia(n_steps, current)
        elif selected == "go":
            if not _ensure_go_loaded():
                raise RuntimeError("Go Cazelles backend is unavailable")
            trace, events, xf = self._simulate_go(n_steps, current)
        elif selected == "mojo":
            if not _ensure_mojo_loaded():
                raise RuntimeError("Mojo Cazelles backend is unavailable")
            trace, events, xf = self._simulate_mojo(n_steps, current)
        else:
            trace, events, xf = self._simulate_python(n_steps, current)
        self.x = xf
        return trace, events

    def _simulate_python(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, float]:
        parameters = self._parameters()
        x = self.x
        trace = np.empty(n_steps, dtype=np.float64)
        events = 0
        for index in range(n_steps):
            candidate = parameters.candidate(x, current)
            events += int(x >= parameters.x1 and candidate < parameters.x1)
            x = candidate
            trace[index] = x
        return trace, events, x

    def _runtime_args(self) -> tuple[float, ...]:
        return self._parameters().float_args(self.x)

    def _simulate_rust(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, float]:
        assert _rust_simulate is not None
        trace, events, xf = _rust_simulate(*self._runtime_args(), self.exponent, n_steps, current)
        return np.asarray(trace, dtype=np.float64), int(events), float(xf)

    def _simulate_julia(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, float]:
        assert _julia_module is not None
        result = _julia_module.simulate_trace(
            *self._runtime_args(), self.exponent, n_steps, current
        )
        return np.asarray(result.trace, dtype=np.float64), int(result.events), float(result.xf)

    def _simulate_go(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, float]:
        assert _go_lib is not None
        trace = np.zeros(n_steps + 1, dtype=np.float64)
        events = _go_lib.cazelles_map_simulate_c(
            *self._runtime_args(),
            ctypes.c_int(self.exponent),
            ctypes.c_int(n_steps),
            ctypes.c_double(current),
            trace.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        if events < 0:
            raise FloatingPointError("Go Cazelles backend rejected the trajectory")
        return np.ascontiguousarray(trace[:n_steps]), int(events), float(trace[n_steps])

    def _simulate_mojo(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, float]:
        assert _mojo_lib is not None
        trace = np.zeros(n_steps + 1, dtype=np.float64)
        events = _mojo_lib.cazelles_map_simulate_c(
            *self._runtime_args(), self.exponent, n_steps, current, int(trace.ctypes.data)
        )
        if events < 0:
            raise FloatingPointError("Mojo Cazelles backend rejected the trajectory")
        return np.ascontiguousarray(trace[:n_steps]), int(events), float(trace[n_steps])

    def reset(self) -> None:
        """Restore Figure 1's initial state while preserving parameters."""
        self.x = 0.1 if self.x0 <= 0.1 <= self.x4 else self.x0
