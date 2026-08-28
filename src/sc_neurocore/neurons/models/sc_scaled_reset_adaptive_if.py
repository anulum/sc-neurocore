# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Retained scaled-reset four-state adaptive integrate-and-fire recurrence

"""Count-neutral project recurrence formerly published as Mihalas-Niebur.

This identity preserves the historical normalised four-state RK4 flow and its
candidate-proportional voltage reset. It intentionally carries no whole-model
publication attribution; the source Mihalaş-Niebur identity lives in
``mihalas_niebur.py``.
"""

from __future__ import annotations

import ctypes
import importlib as _importlib
import importlib.util as _importlib_util
import os as _os
from dataclasses import dataclass, replace
from math import isfinite
from typing import Any, Callable, Optional, cast

import numpy as np
import numpy.typing as npt

_RustSimulate = Callable[..., "tuple[list[float], int, float, float, float, float]"]
_MAX_C_STEPS = (1 << 31) - 1


def _load_rust_simulate() -> _RustSimulate:
    engine = _importlib.import_module("sc_neurocore_engine")
    return cast(_RustSimulate, engine.py_sc_scaled_reset_adaptive_if_simulate)


try:
    _rust_simulate: Optional[_RustSimulate] = _load_rust_simulate()
    _HAS_RUST = True
except (ImportError, AttributeError):
    _rust_simulate = None
    _HAS_RUST = False

_julia_module = None
_go_lib = None
_mojo_lib = None
_ACCEL_ROOT = _os.path.join(_os.path.dirname(__file__), "..", "..", "accel")


def _ensure_julia_loaded() -> bool:
    """Load the retained Julia implementation when available."""
    global _julia_module
    if _julia_module is not None:
        return True
    if _importlib_util.find_spec("juliacall") is None:
        return False
    path = _os.path.abspath(
        _os.path.join(_ACCEL_ROOT, "julia", "neurons", "sc_scaled_reset_adaptive_if.jl")
    )
    if not _os.path.isfile(path):
        return False
    julia = _importlib.import_module("juliacall").Main
    julia.include(path)
    _julia_module = julia.SCScaledResetAdaptiveIFAccel
    return True


def _ensure_go_loaded() -> bool:
    """Load the retained Go C-ABI implementation when available."""
    global _go_lib
    if _go_lib is not None:
        return True
    path = _os.path.abspath(
        _os.path.join(
            _ACCEL_ROOT,
            "go",
            "neurons",
            "sc_scaled_reset_adaptive_if",
            "libsc_scaled_reset_adaptive_if.so",
        )
    )
    if not _os.path.isfile(path):
        return False
    try:
        library = ctypes.CDLL(path)
    except OSError:
        return False
    function = getattr(library, "sc_scaled_reset_adaptive_if_simulate_c", None)
    if function is None:
        return False
    function.argtypes = [ctypes.c_double] * 17 + [
        ctypes.c_int,
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_double),
    ]
    function.restype = ctypes.c_longlong
    _go_lib = library
    return True


def _ensure_mojo_loaded() -> bool:
    """Load the retained Mojo C-ABI implementation when available."""
    global _mojo_lib
    if _mojo_lib is not None:
        return True
    path = _os.path.abspath(
        _os.path.join(_ACCEL_ROOT, "mojo", "neurons", "libsc_scaled_reset_adaptive_if.so")
    )
    if not _os.path.isfile(path):
        return False
    try:
        library = ctypes.CDLL(path)
    except OSError:
        return False
    function = getattr(library, "sc_scaled_reset_adaptive_if_simulate_c", None)
    if function is None:
        return False
    function.argtypes = [ctypes.c_double] * 17 + [ctypes.c_int64, ctypes.c_double, ctypes.c_int64]
    function.restype = ctypes.c_int64
    _mojo_lib = library
    return True


@dataclass
class SCScaledResetAdaptiveIFNeuron:
    """Historical four-state project recurrence with a scaled voltage reset."""

    v: float = 0.0
    theta: float = 1.0
    i1: float = 0.0
    i2: float = 0.0
    v_rest: float = 0.0
    v_reset: float = 0.0
    theta_reset: float = 1.0
    theta_inf: float = 1.0
    tau_v: float = 10.0
    tau_theta: float = 100.0
    tau_1: float = 10.0
    tau_2: float = 200.0
    a: float = 0.0
    b: float = 0.0
    r1: float = 0.0
    r2: float = 0.0
    dt: float = 1.0

    def __post_init__(self) -> None:
        self._raise_if_invalid_runtime()

    def _runtime_args(self) -> tuple[float, ...]:
        return (
            self.v,
            self.theta,
            self.i1,
            self.i2,
            self.v_rest,
            self.v_reset,
            self.theta_reset,
            self.theta_inf,
            self.tau_v,
            self.tau_theta,
            self.tau_1,
            self.tau_2,
            self.a,
            self.b,
            self.r1,
            self.r2,
            self.dt,
        )

    def _raise_if_invalid_runtime(self) -> None:
        for field in (
            "v",
            "theta",
            "i1",
            "i2",
            "v_rest",
            "v_reset",
            "theta_reset",
            "theta_inf",
            "a",
            "b",
            "r1",
            "r2",
        ):
            if not isfinite(getattr(self, field)):
                raise ValueError(f"{field} must be finite")
        for field in ("tau_v", "tau_theta", "tau_1", "tau_2", "dt"):
            value = getattr(self, field)
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"{field} must be finite and positive")

    def _derivatives(
        self, v: float, theta: float, i1: float, i2: float, current: float
    ) -> tuple[float, float, float, float]:
        return (
            (-(v - self.v_rest) + i1 + i2 + current) / self.tau_v,
            (self.theta_inf - theta + self.a * (v - self.v_rest)) / self.tau_theta,
            -i1 / self.tau_1,
            -i2 / self.tau_2,
        )

    @staticmethod
    def _add(
        state: tuple[float, float, float, float],
        slope: tuple[float, float, float, float],
        scale: float,
    ) -> tuple[float, float, float, float]:
        return (
            state[0] + scale * slope[0],
            state[1] + scale * slope[1],
            state[2] + scale * slope[2],
            state[3] + scale * slope[3],
        )

    def _candidate(self, current: float) -> tuple[float, float, float, float]:
        state = (self.v, self.theta, self.i1, self.i2)
        half_dt = 0.5 * self.dt
        k1 = self._derivatives(*state, current)
        k2 = self._derivatives(*self._add(state, k1, half_dt), current)
        k3 = self._derivatives(*self._add(state, k2, half_dt), current)
        k4 = self._derivatives(*self._add(state, k3, self.dt), current)
        return (
            state[0] + self.dt * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0,
            state[1] + self.dt * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0,
            state[2] + self.dt * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]) / 6.0,
            state[3] + self.dt * (k1[3] + 2.0 * k2[3] + 2.0 * k3[3] + k4[3]) / 6.0,
        )

    def step(self, current: float) -> int:
        """Advance the retained recurrence and return its level event."""
        if not isfinite(current):
            raise ValueError("current must be finite")
        self._raise_if_invalid_runtime()
        candidate = self._candidate(current)
        if not all(isfinite(value) for value in candidate):
            raise FloatingPointError("retained scaled-reset candidate state became non-finite")
        event = int(candidate[0] >= candidate[1])
        if event:
            next_state = (
                self.v_reset + self.b * (candidate[0] - self.v_rest),
                max(candidate[1], self.theta_reset),
                candidate[2] + self.r1,
                candidate[3] + self.r2,
            )
        else:
            next_state = candidate
        if not all(isfinite(value) for value in next_state):
            raise FloatingPointError("retained scaled-reset state became non-finite")
        self.v, self.theta, self.i1, self.i2 = next_state
        return event

    def simulate(
        self, n_steps: int, current: float = 0.0, backend: str = "auto"
    ) -> tuple[npt.NDArray[np.float64], int]:
        """Advance the retained trajectory through one explicit runtime."""
        if isinstance(n_steps, bool) or not isinstance(n_steps, int):
            raise ValueError("n_steps must be an integer")
        if not 0 <= n_steps <= _MAX_C_STEPS:
            raise ValueError(f"n_steps must be between 0 and {_MAX_C_STEPS}")
        if backend not in ("auto", "rust", "julia", "go", "mojo", "python"):
            raise ValueError(f"backend must be auto/rust/julia/go/mojo/python, got {backend!r}")
        if not isfinite(current):
            raise ValueError("current must be finite")
        self._raise_if_invalid_runtime()
        if backend == "rust" and not _HAS_RUST:
            raise RuntimeError("Rust retained adaptive-IF backend is unavailable")
        if backend == "julia" and not _ensure_julia_loaded():
            raise RuntimeError("Julia retained adaptive-IF backend is unavailable")
        if backend == "go" and not _ensure_go_loaded():
            raise RuntimeError("Go retained adaptive-IF backend is unavailable")
        if backend == "mojo" and not _ensure_mojo_loaded():
            raise RuntimeError("Mojo retained adaptive-IF backend is unavailable")
        if backend == "rust" or (backend == "auto" and _HAS_RUST):
            trace, spikes, state = self._simulate_rust(n_steps, current)
        elif backend == "julia":
            trace, spikes, state = self._simulate_julia(n_steps, current)
        elif backend == "go":
            trace, spikes, state = self._simulate_native(_go_lib, n_steps, current)
        elif backend == "mojo":
            trace, spikes, state = self._simulate_native(_mojo_lib, n_steps, current)
        else:
            trace, spikes, state = self._simulate_python(n_steps, current)
        self._validate_batch_result(trace, spikes, state, n_steps)
        self.v, self.theta, self.i1, self.i2 = state
        return trace, spikes

    def _simulate_python(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, tuple[float, float, float, float]]:
        worker = replace(self)
        trace = np.empty(n_steps, dtype=np.float64)
        spikes = 0
        for index in range(n_steps):
            spikes += worker.step(current)
            trace[index] = worker.v
        return trace, spikes, (worker.v, worker.theta, worker.i1, worker.i2)

    @staticmethod
    def _validate_batch_result(
        trace: npt.NDArray[np.float64],
        events: int,
        state: tuple[float, float, float, float],
        n_steps: int,
    ) -> None:
        if trace.shape != (n_steps,):
            raise FloatingPointError("retained backend returned an invalid trace length")
        if not np.isfinite(trace).all() or not all(isfinite(value) for value in state):
            raise FloatingPointError("retained backend returned a non-finite result")
        if not 0 <= events <= n_steps:
            raise FloatingPointError("retained backend returned an invalid event count")

    def _simulate_rust(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, tuple[float, float, float, float]]:
        rust_simulate = _rust_simulate
        if rust_simulate is None:
            raise RuntimeError("Rust retained adaptive-IF backend is unavailable")
        result = rust_simulate(*self._runtime_args(), n_steps, current)
        trace, spikes, v, theta, i1, i2 = result
        return np.asarray(trace, dtype=np.float64), int(spikes), (v, theta, i1, i2)

    def _simulate_julia(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, tuple[float, float, float, float]]:
        julia_module = _julia_module
        if julia_module is None:
            raise RuntimeError("Julia retained adaptive-IF backend is unavailable")
        result = julia_module.simulate_trace(
            *(float(value) for value in self._runtime_args()), int(n_steps), float(current)
        )
        return (
            np.asarray(result.trace, dtype=np.float64),
            int(result.spikes),
            (float(result.vf), float(result.theta_f), float(result.i1_f), float(result.i2_f)),
        )

    def _simulate_native(
        self, library: Any, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, tuple[float, float, float, float]]:
        if library is None:
            raise RuntimeError("retained native backend is unavailable")
        trace = np.zeros(n_steps + 4, dtype=np.float64, order="C")
        function = library.sc_scaled_reset_adaptive_if_simulate_c
        if library is _go_lib:
            spikes = function(
                *(ctypes.c_double(value) for value in self._runtime_args()),
                ctypes.c_int(n_steps),
                ctypes.c_double(current),
                trace.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            )
        else:
            spikes = function(
                *(float(value) for value in self._runtime_args()),
                int(n_steps),
                float(current),
                int(trace.ctypes.data),
            )
        if spikes < 0:
            raise FloatingPointError("retained native batch rejected an invalid candidate")
        state = tuple(float(value) for value in trace[n_steps : n_steps + 4])
        return (
            np.ascontiguousarray(trace[:n_steps]),
            int(spikes),
            (state[0], state[1], state[2], state[3]),
        )

    def reset(self) -> None:
        """Restore the retained default state without changing parameters."""
        self.v = self.v_rest
        self.theta = self.theta_reset
        self.i1 = 0.0
        self.i2 = 0.0
