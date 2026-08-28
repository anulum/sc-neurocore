# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Retained clipped rational-recovery map

"""Retained project-defined clipped rational-recovery map.

This count-neutral SC identity preserves the recurrence formerly exposed under
the Courbage-Nekorkin name. It has no whole-model publication attribution:

    f(x) = alpha*x                         for x < 0
           alpha*x/(1 + alpha*x)          for x >= 0
    x[n+1] = clip(f(x[n]) + y[n] + current + j)
    y[n+1] = clip(y[n] - beta*(x[n] + 1))

The clip interval is [-clip_bound, clip_bound]. An event is an upward crossing
of x_threshold. Both candidates use the pre-step state and commit atomically.
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
class _SCParameters:
    alpha: float
    beta: float
    j: float
    x_threshold: float
    clip_bound: float

    def float_args(self, x: float, y: float) -> tuple[float, ...]:
        """Return the stable cross-language floating-point ABI order."""
        return x, y, self.alpha, self.beta, self.j, self.x_threshold, self.clip_bound

    def validate(self, x: float, y: float) -> None:
        """Reject non-finite or structurally invalid runtime state."""
        if not all(math.isfinite(value) for value in self.float_args(x, y)):
            raise ValueError("SC rational-recovery state and parameters must be finite")
        if self.alpha <= 0.0 or self.beta <= 0.0 or self.clip_bound <= 0.0:
            raise ValueError("alpha, beta, and clip_bound must be positive")
        if abs(x) > self.clip_bound or abs(y) > self.clip_bound:
            raise ValueError("SC rational-recovery state exceeds clip_bound")

    def candidate(self, x: float, y: float, current: float) -> tuple[float, float]:
        """Evaluate and clip one simultaneous project-map update."""
        field = self.alpha * x if x < 0.0 else self.alpha * x / (1.0 + self.alpha * x)
        x_new = field + y + current + self.j
        y_new = y - self.beta * (x + 1.0)
        if not math.isfinite(x_new) or not math.isfinite(y_new):
            raise FloatingPointError("SC rational-recovery candidate became non-finite")
        return (
            min(self.clip_bound, max(-self.clip_bound, x_new)),
            min(self.clip_bound, max(-self.clip_bound, y_new)),
        )


class _RustSimulate(Protocol):
    def __call__(self, *args: Any) -> tuple[list[float], int, float, float]: ...


class _JuliaResult(Protocol):
    trace: Any
    events: int
    xf: float
    yf: float


class _JuliaAccel(Protocol):
    def simulate_trace(self, *args: Any) -> _JuliaResult: ...


def _load_rust_simulate() -> _RustSimulate:
    engine = importlib.import_module("sc_neurocore_engine")
    return cast(_RustSimulate, engine.py_sc_clipped_rational_recovery_map_simulate)


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
    path = os.path.abspath(
        os.path.join(
            _ACCEL_ROOT,
            "julia",
            "neurons",
            "sc_clipped_rational_recovery_map.jl",
        )
    )
    if not os.path.isfile(path):
        return False
    julia = importlib.import_module("juliacall").Main
    julia.include(path)
    _julia_module = cast(_JuliaAccel, julia.SCClippedRationalRecoveryMapAccel)
    return True


def _load_c_backend(path: str, *, mojo: bool) -> ctypes.CDLL | None:
    if not os.path.isfile(path):
        return None
    try:
        library = ctypes.CDLL(path)
    except OSError:
        return None
    function = getattr(library, "sc_clipped_rational_recovery_map_simulate_c", None)
    if function is None:
        return None
    if mojo:
        function.argtypes = [ctypes.c_double] * 7 + [
            ctypes.c_int64,
            ctypes.c_double,
            ctypes.c_int64,
        ]
    else:
        function.argtypes = [ctypes.c_double] * 7 + [
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
            "sc_clipped_rational_recovery_map",
            "libsc_clipped_rational_recovery_map.so",
        )
    )
    _go_lib = _load_c_backend(path, mojo=False)
    return _go_lib is not None


def _ensure_mojo_loaded() -> bool:
    global _mojo_lib
    if _mojo_lib is not None:
        return True
    path = os.path.abspath(
        os.path.join(
            _ACCEL_ROOT,
            "mojo",
            "neurons",
            "libsc_clipped_rational_recovery_map.so",
        )
    )
    _mojo_lib = _load_c_backend(path, mojo=True)
    return _mojo_lib is not None


@dataclass
class SCClippedRationalRecoveryMapNeuron:
    """Retained clipped rational-recovery project recurrence."""

    x: float = 0.0
    y: float = 0.0
    alpha: float = 3.0
    beta: float = 0.001
    j: float = 0.1
    x_threshold: float = 1.0
    clip_bound: float = 1_000_000.0

    def __post_init__(self) -> None:
        self._parameters().validate(self.x, self.y)

    def _parameters(self) -> _SCParameters:
        return _SCParameters(
            self.alpha,
            self.beta,
            self.j,
            self.x_threshold,
            self.clip_bound,
        )

    def _f(self, x: float) -> float:
        """Evaluate the retained piecewise rational fast map."""
        return self.alpha * x if x < 0.0 else self.alpha * x / (1.0 + self.alpha * x)

    def step(self, current: float = 0.0) -> int:
        """Advance once; rejected candidates leave both states unchanged."""
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
                raise RuntimeError("Rust SC rational-recovery backend is unavailable")
            trace, events, x_final, y_final = self._simulate_rust(n_steps, current)
        elif selected == "julia":
            if not _ensure_julia_loaded():
                raise RuntimeError("Julia SC rational-recovery backend is unavailable")
            trace, events, x_final, y_final = self._simulate_julia(n_steps, current)
        elif selected == "go":
            if not _ensure_go_loaded():
                raise RuntimeError("Go SC rational-recovery backend is unavailable")
            trace, events, x_final, y_final = self._simulate_go(n_steps, current)
        elif selected == "mojo":
            if not _ensure_mojo_loaded():
                raise RuntimeError("Mojo SC rational-recovery backend is unavailable")
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
            int(result.events),
            float(result.xf),
            float(result.yf),
        )

    def _simulate_go(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, float, float]:
        assert _go_lib is not None
        trace = np.zeros(n_steps + 2, dtype=np.float64, order="C")
        events = _go_lib.sc_clipped_rational_recovery_map_simulate_c(
            *(ctypes.c_double(value) for value in self._runtime_args()),
            ctypes.c_int(n_steps),
            ctypes.c_double(current),
            trace.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        if events < 0:
            raise FloatingPointError("Go SC rational-recovery backend rejected the trajectory")
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
        events = _mojo_lib.sc_clipped_rational_recovery_map_simulate_c(
            *self._runtime_args(), n_steps, current, int(trace.ctypes.data)
        )
        if events < 0:
            raise FloatingPointError("Mojo SC rational-recovery backend rejected the trajectory")
        return (
            np.ascontiguousarray(trace[:n_steps]),
            int(events),
            float(trace[-2]),
            float(trace[-1]),
        )

    def reset(self) -> None:
        """Restore the retained project initial state."""
        self.x = 0.0
        self.y = 0.0
