# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Teeter 2018 GLIF5 source model

r"""Teeter et al. (2018) five-state GLIF5 point neuron.

The source states are membrane voltage :math:`V`, spike-dependent threshold
:math:`\Theta_s`, two after-spike currents :math:`I_1,I_2`, and the
voltage-dependent threshold component :math:`\Theta_v`. Between events,

.. math::

   V' = (I_e + I_1 + I_2 - (V-E_L)/R)/C,

.. math::

   \Theta_s'=-b_s\Theta_s,\quad I_j'=-k_jI_j,\quad
   \Theta_v'=a_v(V-E_L)-b_v\Theta_v.

An event requires the strict source condition
``V > theta_inf + theta_spike + theta_voltage``. The fitted affine voltage,
threshold, and after-spike-current resets are applied to the finite candidate;
the returned state is the post-cut state and ``refractory_remaining`` records
the input-suppression interval. The historical four-state RK4 project
recurrence remains available as :class:`SCFourStateGLIFNeuron`.

Reference
---------
Teeter, C. et al. (2018). *Generalized leaky integrate-and-fire models
classify multiple neuron types*. Nature Communications 9, 709.
https://doi.org/10.1038/s41467-017-02717-4
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

from sc_neurocore.accel.backend_order import with_floor
from sc_neurocore.accel.backend_selection import select_backend_order

_MAX_C_STEPS = (1 << 31) - 1
_AUTO_BACKENDS = with_floor("python")
_BENCHMARK_KERNEL = "glif_simulate"
_ACCEL_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "accel")
_FLOAT_ARGUMENTS = 24


def _decay(rate: float, dt: float) -> float:
    """Return ``exp(-rate * dt)`` for a validated positive rate and step."""
    return math.exp(-rate * dt)


def _exponential_convolution(decay_rate: float, forcing_rate: float, dt: float) -> float:
    """Integrate one exponential forcing through a linear decay exactly."""
    difference = decay_rate - forcing_rate
    scale = max(1.0, abs(decay_rate), abs(forcing_rate))
    if abs(difference) <= 1e-12 * scale:
        return dt * math.exp(-decay_rate * dt)
    return (math.exp(-forcing_rate * dt) - math.exp(-decay_rate * dt)) / difference


@dataclass(frozen=True, slots=True)
class _GLIF5Parameters:
    """Validated source parameters in stable cross-language ABI order."""

    e_l: float
    capacitance: float
    resistance: float
    theta_inf: float
    b_spike: float
    b_voltage: float
    a_voltage: float
    k_asc1: float
    k_asc2: float
    f_v: float
    delta_v: float
    delta_theta_spike: float
    f_asc1: float
    f_asc2: float
    delta_i_asc1: float
    delta_i_asc2: float
    refractory_period: float
    dt: float

    def as_tuple(self) -> tuple[float, ...]:
        """Return parameters in the stable native ABI order."""
        return (
            self.e_l,
            self.capacitance,
            self.resistance,
            self.theta_inf,
            self.b_spike,
            self.b_voltage,
            self.a_voltage,
            self.k_asc1,
            self.k_asc2,
            self.f_v,
            self.delta_v,
            self.delta_theta_spike,
            self.f_asc1,
            self.f_asc2,
            self.delta_i_asc1,
            self.delta_i_asc2,
            self.refractory_period,
            self.dt,
        )


@dataclass(frozen=True, slots=True)
class _GLIF5State:
    """One complete GLIF5 runtime state."""

    v: float
    theta_spike: float
    i_asc1: float
    i_asc2: float
    theta_voltage: float
    refractory_remaining: float

    def as_tuple(self) -> tuple[float, ...]:
        """Return state in the stable native ABI order."""
        return (
            self.v,
            self.theta_spike,
            self.i_asc1,
            self.i_asc2,
            self.theta_voltage,
            self.refractory_remaining,
        )

    def candidate(self, current: float, parameters: _GLIF5Parameters) -> tuple[_GLIF5State, int]:
        """Evaluate one source-faithful exact-flow interval without mutation."""
        if self.refractory_remaining > 0.0:
            remaining = max(0.0, self.refractory_remaining - parameters.dt)
            return _GLIF5State(
                self.v,
                self.theta_spike,
                self.i_asc1,
                self.i_asc2,
                self.theta_voltage,
                remaining,
            ), 0

        total_current = current + self.i_asc1 + self.i_asc2
        membrane_rate = 1.0 / (parameters.resistance * parameters.capacitance)
        membrane_decay = _decay(membrane_rate, parameters.dt)
        equilibrium_offset = parameters.resistance * total_current
        voltage_offset = self.v - parameters.e_l
        next_offset = equilibrium_offset + (voltage_offset - equilibrium_offset) * membrane_decay
        next_v = parameters.e_l + next_offset

        next_theta_spike = self.theta_spike * _decay(parameters.b_spike, parameters.dt)
        next_i_asc1 = self.i_asc1 * _decay(parameters.k_asc1, parameters.dt)
        next_i_asc2 = self.i_asc2 * _decay(parameters.k_asc2, parameters.dt)
        threshold_forcing = equilibrium_offset * (
            1.0 - _decay(parameters.b_voltage, parameters.dt)
        ) / parameters.b_voltage + (voltage_offset - equilibrium_offset) * _exponential_convolution(
            parameters.b_voltage, membrane_rate, parameters.dt
        )
        next_theta_voltage = (
            self.theta_voltage * _decay(parameters.b_voltage, parameters.dt)
            + parameters.a_voltage * threshold_forcing
        )
        candidate = _GLIF5State(
            next_v,
            next_theta_spike,
            next_i_asc1,
            next_i_asc2,
            next_theta_voltage,
            0.0,
        )
        if not all(math.isfinite(value) for value in candidate.as_tuple()):
            raise FloatingPointError("GLIF5 candidate state became non-finite")

        threshold = parameters.theta_inf + next_theta_spike + next_theta_voltage
        if next_v <= threshold:
            return candidate, 0
        reset = _GLIF5State(
            parameters.e_l + parameters.f_v * (next_v - parameters.e_l) - parameters.delta_v,
            next_theta_spike + parameters.delta_theta_spike,
            parameters.f_asc1 * next_i_asc1 + parameters.delta_i_asc1,
            parameters.f_asc2 * next_i_asc2 + parameters.delta_i_asc2,
            next_theta_voltage,
            parameters.refractory_period,
        )
        if not all(math.isfinite(value) for value in reset.as_tuple()):
            raise FloatingPointError("GLIF5 reset state became non-finite")
        return reset, 1


class _RustSimulate(Protocol):
    """Callable surface exported by the production Rust engine."""

    def __call__(
        self, *args: float | int
    ) -> tuple[Any, int, float, float, float, float, float, float]: ...


class _JuliaResult(Protocol):
    """Shape returned by the Julia batch function."""

    trace: Any
    events: int
    vf: float
    theta_spike_f: float
    i_asc1_f: float
    i_asc2_f: float
    theta_voltage_f: float
    refractory_f: float


class _JuliaAccel(Protocol):
    """Callable surface exposed by the loaded Julia module."""

    def simulate_trace(self, *args: float | int) -> _JuliaResult: ...


def _load_rust_simulate() -> _RustSimulate:
    engine = importlib.import_module("sc_neurocore_engine")
    return cast(_RustSimulate, engine.py_glif_simulate)


try:
    _rust_simulate: _RustSimulate | None = _load_rust_simulate()
    _HAS_RUST = True
except (ImportError, AttributeError):
    _rust_simulate = None
    _HAS_RUST = False

_julia_module: _JuliaAccel | None = None
_go_lib: ctypes.CDLL | None = None
_mojo_lib: ctypes.CDLL | None = None


def _load_c_backend(path: str, *, mojo: bool) -> ctypes.CDLL | None:
    if not os.path.isfile(path):
        return None
    try:
        library = ctypes.CDLL(path)
    except OSError:
        return None
    function = getattr(library, "glif_simulate_c", None)
    if function is None:
        return None
    if mojo:
        function.argtypes = [ctypes.c_double] * _FLOAT_ARGUMENTS + [
            ctypes.c_int64,
            ctypes.c_double,
            ctypes.c_int64,
        ]
    else:
        function.argtypes = [ctypes.c_double] * _FLOAT_ARGUMENTS + [
            ctypes.c_int,
            ctypes.c_double,
            ctypes.POINTER(ctypes.c_double),
        ]
    function.restype = ctypes.c_longlong
    return library


def _ensure_julia_loaded() -> bool:
    global _julia_module
    if _julia_module is not None:
        return True
    if importlib.util.find_spec("juliacall") is None:
        return False
    path = os.path.abspath(os.path.join(_ACCEL_ROOT, "julia", "neurons", "glif.jl"))
    if not os.path.isfile(path):
        return False
    juliacall = importlib.import_module("juliacall")
    julia = juliacall.Main
    julia.include(path)
    _julia_module = cast(_JuliaAccel, julia.GLIF5Accel)
    return True


def _ensure_go_loaded() -> bool:
    global _go_lib
    if _go_lib is None:
        path = os.path.abspath(os.path.join(_ACCEL_ROOT, "go", "neurons", "glif", "libglif.so"))
        _go_lib = _load_c_backend(path, mojo=False)
    return _go_lib is not None


def _ensure_mojo_loaded() -> bool:
    global _mojo_lib
    if _mojo_lib is None:
        path = os.path.abspath(os.path.join(_ACCEL_ROOT, "mojo", "neurons", "libglif.so"))
        _mojo_lib = _load_c_backend(path, mojo=True)
    return _mojo_lib is not None


def _backend_available(backend: str) -> bool:
    if backend == "rust":
        return _HAS_RUST
    if backend == "julia":
        return _ensure_julia_loaded()
    if backend == "go":
        return _ensure_go_loaded()
    if backend == "mojo":
        return _ensure_mojo_loaded()
    return backend == "python"


def _auto_backend() -> str:
    ordered = select_backend_order(_BENCHMARK_KERNEL, static=_AUTO_BACKENDS)
    return next((backend for backend in ordered if _backend_available(backend)), "python")


@dataclass
class GLIFNeuron:
    """Teeter et al. GLIF5 with five source states and fitted reset rules.

    The defaults form a source-consistent normalised operating profile. They
    are not attributed to one Allen Cell Types Atlas specimen. ``current`` is
    constant over each exact-flow interval and uses the same current units as
    ``i_asc1`` and ``i_asc2``.
    """

    v: float = -70.0
    theta_spike: float = 0.0
    i_asc1: float = 0.0
    i_asc2: float = 0.0
    theta_voltage: float = 0.0
    refractory_remaining: float = 0.0
    e_l: float = -70.0
    capacitance: float = 10.0
    resistance: float = 1.0
    theta_inf: float = -50.0
    b_spike: float = 0.01
    b_voltage: float = 0.01
    a_voltage: float = 0.0001
    k_asc1: float = 0.1
    k_asc2: float = 0.005
    f_v: float = 0.0
    delta_v: float = 0.0
    delta_theta_spike: float = 2.0
    f_asc1: float = 1.0
    f_asc2: float = 1.0
    delta_i_asc1: float = 1.0
    delta_i_asc2: float = 0.5
    refractory_period: float = 2.0
    dt: float = 1.0

    def __post_init__(self) -> None:
        self._state()
        self._parameters()

    @property
    def theta(self) -> float:
        """Return the instantaneous composite spike threshold."""
        return self.theta_inf + self.theta_spike + self.theta_voltage

    def _state(self) -> _GLIF5State:
        state = _GLIF5State(
            float(self.v),
            float(self.theta_spike),
            float(self.i_asc1),
            float(self.i_asc2),
            float(self.theta_voltage),
            float(self.refractory_remaining),
        )
        if not all(math.isfinite(value) for value in state.as_tuple()):
            raise ValueError("GLIF5 state must be finite")
        if state.refractory_remaining < 0.0:
            raise ValueError("refractory_remaining must be non-negative")
        return state

    def _parameters(self) -> _GLIF5Parameters:
        parameters = _GLIF5Parameters(
            float(self.e_l),
            float(self.capacitance),
            float(self.resistance),
            float(self.theta_inf),
            float(self.b_spike),
            float(self.b_voltage),
            float(self.a_voltage),
            float(self.k_asc1),
            float(self.k_asc2),
            float(self.f_v),
            float(self.delta_v),
            float(self.delta_theta_spike),
            float(self.f_asc1),
            float(self.f_asc2),
            float(self.delta_i_asc1),
            float(self.delta_i_asc2),
            float(self.refractory_period),
            float(self.dt),
        )
        if not all(math.isfinite(value) for value in parameters.as_tuple()):
            raise ValueError("GLIF5 parameters must be finite")
        for name in (
            "capacitance",
            "resistance",
            "b_spike",
            "b_voltage",
            "k_asc1",
            "k_asc2",
            "dt",
        ):
            if getattr(parameters, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if parameters.refractory_period < 0.0:
            raise ValueError("refractory_period must be non-negative")
        return parameters

    def _commit(self, state: _GLIF5State) -> None:
        (
            self.v,
            self.theta_spike,
            self.i_asc1,
            self.i_asc2,
            self.theta_voltage,
            self.refractory_remaining,
        ) = state.as_tuple()

    def step(self, current: float = 0.0) -> int:
        """Advance one exact-flow interval and return the strict GLIF5 event."""
        drive = float(current)
        if not math.isfinite(drive):
            raise ValueError("current must be finite")
        candidate, event = self._state().candidate(drive, self._parameters())
        self._commit(candidate)
        return event

    def simulate(
        self, n_steps: int, current: float = 0.0, backend: str = "auto"
    ) -> tuple[npt.NDArray[np.float64], int]:
        """Run a failure-atomic constant-current batch and return voltage/events."""
        if isinstance(n_steps, bool) or not isinstance(n_steps, int):
            raise ValueError("n_steps must be an integer")
        if not 0 <= n_steps <= _MAX_C_STEPS:
            raise ValueError(f"n_steps must be between 0 and {_MAX_C_STEPS}")
        drive = float(current)
        if not math.isfinite(drive):
            raise ValueError("current must be finite")
        if backend not in {"auto", "rust", "julia", "go", "mojo", "python"}:
            raise ValueError(
                f"backend must be one of auto/rust/julia/go/mojo/python, got {backend!r}"
            )
        state = self._state()
        parameters = self._parameters()
        selected = _auto_backend() if backend == "auto" else backend
        if selected != "python" and not _backend_available(selected):
            raise RuntimeError(f"{selected} GLIF5 backend is unavailable")
        if selected == "rust":
            result = self._simulate_rust(n_steps, drive, state, parameters)
        elif selected == "julia":
            result = self._simulate_julia(n_steps, drive, state, parameters)
        elif selected == "go":
            result = self._simulate_c(_go_lib, "go", n_steps, drive, state, parameters, mojo=False)
        elif selected == "mojo":
            result = self._simulate_c(
                _mojo_lib, "mojo", n_steps, drive, state, parameters, mojo=True
            )
        else:
            result = self._simulate_python(n_steps, drive, state, parameters)
        trace, events, final_state = result
        self._commit(final_state)
        return trace, events

    @staticmethod
    def _simulate_python(
        n_steps: int,
        current: float,
        state: _GLIF5State,
        parameters: _GLIF5Parameters,
    ) -> tuple[npt.NDArray[np.float64], int, _GLIF5State]:
        trace = np.empty(n_steps, dtype=np.float64)
        events = 0
        for index in range(n_steps):
            state, event = state.candidate(current, parameters)
            trace[index] = state.v
            events += event
        return trace, events, state

    @staticmethod
    def _native_args(state: _GLIF5State, parameters: _GLIF5Parameters) -> tuple[float, ...]:
        return state.as_tuple() + parameters.as_tuple()

    def _simulate_rust(
        self,
        n_steps: int,
        current: float,
        state: _GLIF5State,
        parameters: _GLIF5Parameters,
    ) -> tuple[npt.NDArray[np.float64], int, _GLIF5State]:
        if _rust_simulate is None:
            raise RuntimeError("rust GLIF5 backend is unavailable")
        result = _rust_simulate(*self._native_args(state, parameters), n_steps, current)
        trace, events, *final_values = result
        return (
            np.asarray(trace, dtype=np.float64),
            int(events),
            _GLIF5State(*map(float, final_values)),
        )

    def _simulate_julia(
        self,
        n_steps: int,
        current: float,
        state: _GLIF5State,
        parameters: _GLIF5Parameters,
    ) -> tuple[npt.NDArray[np.float64], int, _GLIF5State]:
        if _julia_module is None:
            raise RuntimeError("julia GLIF5 backend is unavailable")
        try:
            result = _julia_module.simulate_trace(
                *self._native_args(state, parameters), n_steps, current
            )
        except Exception as error:
            if (
                error.__class__.__module__ != "juliacall"
                or error.__class__.__name__ != "JuliaError"
            ):
                raise
            raise FloatingPointError("Julia GLIF5 backend rejected an invalid state") from error
        final_state = _GLIF5State(
            float(result.vf),
            float(result.theta_spike_f),
            float(result.i_asc1_f),
            float(result.i_asc2_f),
            float(result.theta_voltage_f),
            float(result.refractory_f),
        )
        return np.asarray(result.trace, dtype=np.float64), int(result.events), final_state

    def _simulate_c(
        self,
        library: ctypes.CDLL | None,
        label: str,
        n_steps: int,
        current: float,
        state: _GLIF5State,
        parameters: _GLIF5Parameters,
        *,
        mojo: bool,
    ) -> tuple[npt.NDArray[np.float64], int, _GLIF5State]:
        if library is None:
            raise RuntimeError(f"{label} GLIF5 backend is unavailable")
        trace = np.zeros(n_steps + 6, dtype=np.float64, order="C")
        function = library.glif_simulate_c
        if mojo:
            events = function(
                *self._native_args(state, parameters), n_steps, current, int(trace.ctypes.data)
            )
        else:
            events = function(
                *(ctypes.c_double(value) for value in self._native_args(state, parameters)),
                ctypes.c_int(n_steps),
                ctypes.c_double(current),
                trace.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            )
        if events < 0:
            raise FloatingPointError(f"{label} GLIF5 backend rejected an invalid state")
        final_state = _GLIF5State(*(float(value) for value in trace[n_steps : n_steps + 6]))
        return np.ascontiguousarray(trace[:n_steps]), int(events), final_state

    def reset(self) -> None:
        """Restore the source-consistent operating-profile state."""
        self.v = self.e_l
        self.theta_spike = 0.0
        self.i_asc1 = 0.0
        self.i_asc2 = 0.0
        self.theta_voltage = 0.0
        self.refractory_remaining = 0.0


__all__ = ["GLIFNeuron"]
