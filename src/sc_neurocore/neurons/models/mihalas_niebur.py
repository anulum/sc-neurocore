# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source-faithful Mihalas-Niebur generalized integrate-and-fire model

"""Mihalaş-Niebur (2009) generalized linear integrate-and-fire neuron.

The current states and input are normalised by membrane capacitance, so they
have units of voltage per time. The remaining coefficients map directly to
equations (2.1) and (2.2) of doi:10.1162/neco.2008.12-07-680. A fixed-grid RK4
step with sampled threshold detection is the declared numerical
specialisation; the paper's continuous flow and event reset are unchanged.
"""

from __future__ import annotations

import importlib as _importlib
import os as _os
from dataclasses import dataclass, replace
from math import isfinite
from typing import Callable, Optional, cast

import numpy as np
import numpy.typing as npt

_RustSimulate = Callable[..., "tuple[list[float], int, float, float, float, float]"]
_MAX_C_STEPS = (1 << 31) - 1


def _load_rust_simulate() -> _RustSimulate:
    engine = _importlib.import_module("sc_neurocore_engine")
    return cast(_RustSimulate, engine.py_mihalas_niebur_simulate)


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
    """Load the Julia runtime module when its optional bridge is available."""
    global _julia_module, _HAS_JULIA
    if _julia_module is not None:
        return True
    import importlib.util as importlib_util

    if importlib_util.find_spec("juliacall") is None:
        return False
    path = _os.path.abspath(_os.path.join(_ACCEL_ROOT, "julia", "neurons", "mihalas_niebur.jl"))
    if not _os.path.isfile(path):
        return False
    juliacall = _importlib.import_module("juliacall")
    julia = juliacall.Main
    julia.include(path)
    _julia_module = julia.MihalasNieburAccel
    _HAS_JULIA = True
    return True


def _ensure_go_loaded() -> bool:
    """Load the Go C-ABI library when the focused build has produced it."""
    global _go_lib, _HAS_GO
    if _go_lib is not None:
        return True
    import ctypes

    path = _os.path.abspath(
        _os.path.join(_ACCEL_ROOT, "go", "neurons", "mihalas_niebur", "libmihalasniebur.so")
    )
    if not _os.path.isfile(path):
        return False
    try:
        library = ctypes.CDLL(path)
    except OSError:
        return False
    function = getattr(library, "mihalas_niebur_simulate_c", None)
    if function is None:
        return False
    function.argtypes = [ctypes.c_double] * 18 + [
        ctypes.c_int,
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_double),
    ]
    function.restype = ctypes.c_longlong
    _go_lib = library
    _HAS_GO = True
    return True


def _ensure_mojo_loaded() -> bool:
    """Load the Mojo C-ABI library when the focused build has produced it."""
    global _mojo_lib, _HAS_MOJO
    if _mojo_lib is not None:
        return True
    import ctypes

    path = _os.path.abspath(_os.path.join(_ACCEL_ROOT, "mojo", "neurons", "libmihalasniebur.so"))
    if not _os.path.isfile(path):
        return False
    try:
        library = ctypes.CDLL(path)
    except OSError:
        return False
    function = getattr(library, "mihalas_niebur_simulate_c", None)
    if function is None:
        return False
    function.argtypes = [ctypes.c_double] * 18 + [ctypes.c_int64, ctypes.c_double, ctypes.c_int64]
    function.restype = ctypes.c_int64
    _mojo_lib = library
    _HAS_MOJO = True
    return True


@dataclass
class MihalasNieburNeuron:
    """Source-form Mihalaş-Niebur generalized linear IF neuron.

    Defaults use the paper's common Figure 1 constants and the Figure 1C
    spike-frequency-adaptation value ``a = 5 s⁻¹``. Rates are expressed per
    millisecond, voltages in volts, and ``i1``, ``i2``, and ``current`` in volts
    per millisecond after division by capacitance.

    Reference
    ---------
    Mihalaş, Ş. and Niebur, E. (2009), Neural Computation 21(3), 704–718,
    doi:10.1162/neco.2008.12-07-680, equations (2.1)–(2.2), Table 1.
    """

    v: float = -0.07
    theta: float = -0.05
    i1: float = 0.0
    i2: float = 0.0
    v_rest: float = -0.07
    v_reset: float = -0.07
    theta_reset: float = -0.06
    theta_inf: float = -0.05
    leak_rate: float = 0.05
    threshold_voltage_coupling: float = 0.005
    threshold_decay_rate: float = 0.01
    current_decay_rate_1: float = 0.2
    current_decay_rate_2: float = 0.02
    current_retention_1: float = 0.0
    current_retention_2: float = 1.0
    current_jump_1: float = 0.0
    current_jump_2: float = 0.0
    dt: float = 0.1

    def __post_init__(self) -> None:
        self._raise_if_invalid_runtime()

    @staticmethod
    def _finite_values(values: tuple[float, ...]) -> bool:
        return all(isfinite(value) for value in values)

    def _raise_if_invalid_runtime(self) -> None:
        finite_fields = (
            "v",
            "theta",
            "i1",
            "i2",
            "v_rest",
            "v_reset",
            "theta_reset",
            "theta_inf",
            "threshold_voltage_coupling",
            "current_retention_1",
            "current_retention_2",
            "current_jump_1",
            "current_jump_2",
        )
        for field in finite_fields:
            if not isfinite(getattr(self, field)):
                raise ValueError(f"{field} must be finite")
        for field in (
            "leak_rate",
            "threshold_decay_rate",
            "current_decay_rate_1",
            "current_decay_rate_2",
            "dt",
        ):
            value = getattr(self, field)
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"{field} must be finite and positive")
        if self.theta_reset <= self.v_reset:
            raise ValueError("theta_reset must exceed v_reset as required by equation 2.2")

    def _state_and_parameters(self) -> tuple[float, ...]:
        return (
            self.v,
            self.theta,
            self.i1,
            self.i2,
            self.v_rest,
            self.v_reset,
            self.theta_reset,
            self.theta_inf,
            self.leak_rate,
            self.threshold_voltage_coupling,
            self.threshold_decay_rate,
            self.current_decay_rate_1,
            self.current_decay_rate_2,
            self.current_retention_1,
            self.current_retention_2,
            self.current_jump_1,
            self.current_jump_2,
            self.dt,
        )

    def _derivatives(
        self,
        v: float,
        theta: float,
        i1: float,
        i2: float,
        current: float,
    ) -> tuple[float, float, float, float]:
        return (
            current + i1 + i2 - self.leak_rate * (v - self.v_rest),
            self.threshold_voltage_coupling * (v - self.v_rest)
            - self.threshold_decay_rate * (theta - self.theta_inf),
            -self.current_decay_rate_1 * i1,
            -self.current_decay_rate_2 * i2,
        )

    @staticmethod
    def _add_scaled(
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

    def _rk4_candidate(self, current: float) -> tuple[float, float, float, float]:
        state = (self.v, self.theta, self.i1, self.i2)
        half_dt = 0.5 * self.dt
        k1 = self._derivatives(*state, current)
        k2 = self._derivatives(*self._add_scaled(state, k1, half_dt), current)
        k3 = self._derivatives(*self._add_scaled(state, k2, half_dt), current)
        k4 = self._derivatives(*self._add_scaled(state, k3, self.dt), current)
        return (
            state[0] + self.dt * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0,
            state[1] + self.dt * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0,
            state[2] + self.dt * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]) / 6.0,
            state[3] + self.dt * (k1[3] + 2.0 * k2[3] + 2.0 * k3[3] + k4[3]) / 6.0,
        )

    def step(self, current: float) -> int:
        """Advance one sampled RK4 interval and return one on a source event."""
        if not isfinite(current):
            raise ValueError("current must be finite")
        self._raise_if_invalid_runtime()
        candidate = self._rk4_candidate(current)
        if not self._finite_values(candidate):
            raise FloatingPointError("Mihalas-Niebur candidate state became non-finite")
        event = int(candidate[0] >= candidate[1])
        if event:
            next_state = (
                self.v_reset,
                max(self.theta_reset, candidate[1]),
                self.current_retention_1 * candidate[2] + self.current_jump_1,
                self.current_retention_2 * candidate[3] + self.current_jump_2,
            )
        else:
            next_state = candidate
        if not self._finite_values(next_state):
            raise FloatingPointError("Mihalas-Niebur reset state became non-finite")
        self.v, self.theta, self.i1, self.i2 = next_state
        return event

    def simulate(
        self, n_steps: int, current: float = 0.0, backend: str = "auto"
    ) -> tuple[npt.NDArray[np.float64], int]:
        """Advance a constant-current trajectory through the selected runtime."""
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
            raise RuntimeError("Rust Mihalas-Niebur backend is unavailable")
        if backend == "julia" and not _ensure_julia_loaded():
            raise RuntimeError("Julia Mihalas-Niebur backend is unavailable")
        if backend == "go" and not _ensure_go_loaded():
            raise RuntimeError(
                "Go Mihalas-Niebur backend is unavailable; build libmihalasniebur.so"
            )
        if backend == "mojo" and not _ensure_mojo_loaded():
            raise RuntimeError(
                "Mojo Mihalas-Niebur backend is unavailable; build its shared library"
            )

        if backend == "rust" or (backend == "auto" and _HAS_RUST):
            trace, spikes, state = self._simulate_rust(n_steps, current)
        elif backend == "julia":
            trace, spikes, state = self._simulate_julia(n_steps, current)
        elif backend == "go":
            trace, spikes, state = self._simulate_go(n_steps, current)
        elif backend == "mojo":
            trace, spikes, state = self._simulate_mojo(n_steps, current)
        else:
            trace, spikes, state = self._simulate_python(n_steps, current)
        self._validate_batch_result(trace, spikes, state, n_steps)
        self.v, self.theta, self.i1, self.i2 = state
        return trace, spikes

    @staticmethod
    def _validate_batch_result(
        trace: npt.NDArray[np.float64],
        events: int,
        state: tuple[float, float, float, float],
        n_steps: int,
    ) -> None:
        if trace.shape != (n_steps,):
            raise FloatingPointError("Mihalas-Niebur backend returned an invalid trace length")
        if not np.isfinite(trace).all() or not MihalasNieburNeuron._finite_values(state):
            raise FloatingPointError("Mihalas-Niebur backend returned a non-finite result")
        if not 0 <= events <= n_steps:
            raise FloatingPointError("Mihalas-Niebur backend returned an invalid event count")

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

    def _runtime_args(self) -> tuple[float, ...]:
        return self._state_and_parameters()

    def _simulate_rust(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, tuple[float, float, float, float]]:
        rust_simulate = _rust_simulate
        if rust_simulate is None:
            raise RuntimeError("Rust Mihalas-Niebur backend is unavailable")
        trace, spikes, v, theta, i1, i2 = rust_simulate(*self._runtime_args(), n_steps, current)
        return np.asarray(trace, dtype=np.float64), int(spikes), (v, theta, i1, i2)

    def _simulate_julia(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, tuple[float, float, float, float]]:
        julia_module = _julia_module
        if julia_module is None:
            raise RuntimeError("Julia Mihalas-Niebur backend is unavailable")
        result = julia_module.simulate_trace(
            *(float(value) for value in self._runtime_args()), int(n_steps), float(current)
        )
        return (
            np.asarray(result.trace, dtype=np.float64),
            int(result.spikes),
            (float(result.vf), float(result.theta_f), float(result.i1_f), float(result.i2_f)),
        )

    def _simulate_go(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, tuple[float, float, float, float]]:
        library = _go_lib
        if library is None:
            raise RuntimeError("Go Mihalas-Niebur backend is unavailable")
        import ctypes

        trace = np.zeros(n_steps + 4, dtype=np.float64, order="C")
        spikes = library.mihalas_niebur_simulate_c(
            *(ctypes.c_double(value) for value in self._runtime_args()),
            ctypes.c_int(n_steps),
            ctypes.c_double(current),
            trace.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        if spikes < 0:
            raise FloatingPointError("Go Mihalas-Niebur batch rejected an invalid candidate")
        state = self._native_final_state(trace, n_steps)
        return np.ascontiguousarray(trace[:n_steps]), int(spikes), state

    def _simulate_mojo(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, tuple[float, float, float, float]]:
        library = _mojo_lib
        if library is None:
            raise RuntimeError("Mojo Mihalas-Niebur backend is unavailable")
        trace = np.zeros(n_steps + 4, dtype=np.float64, order="C")
        spikes = library.mihalas_niebur_simulate_c(
            *(float(value) for value in self._runtime_args()),
            int(n_steps),
            float(current),
            int(trace.ctypes.data),
        )
        if spikes < 0:
            raise FloatingPointError("Mojo Mihalas-Niebur batch rejected an invalid candidate")
        state = self._native_final_state(trace, n_steps)
        return np.ascontiguousarray(trace[:n_steps]), int(spikes), state

    def _native_final_state(
        self, trace: npt.NDArray[np.float64], n_steps: int
    ) -> tuple[float, float, float, float]:
        if n_steps == 0:
            return self.v, self.theta, self.i1, self.i2
        return (
            float(trace[n_steps]),
            float(trace[n_steps + 1]),
            float(trace[n_steps + 2]),
            float(trace[n_steps + 3]),
        )

    def reset(self) -> None:
        """Restore the paper-profile resting state without changing parameters."""
        self.v = self.v_rest
        self.theta = self.theta_inf
        self.i1 = 0.0
        self.i2 = 0.0
