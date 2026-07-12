# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Medvedev 2005 slow-calcium first-return map

"""Source-derived Medvedev slow-calcium first-return map.

Medvedev reduces a conductance-based excitable-cell system to a scalar
first-return map ``u[n+1] = P(u[n])`` for the slow calcium variable.  The
maintained recurrence follows the three asymptotic regions constructed in
Section 4 of the source:

* the active left branch uses the exponential-relaxation form of Eq. 4.4
  (Eq. 4.7 is its leading small-alpha form);
* the inner branch composes Eq. 4.8 with the homoclinic boundary layer of
  Eq. 4.13; and
* the slow right branch returns exactly to ``u_SN`` as in Eq. 4.15.

The paper specifies the asymptotic construction but does not tabulate unique
global functions ``T(u)`` and ``F(u)``.  Consequently, ``decay_t0``, ``f_0``,
``f_1``, ``homoclinic_exponent`` and ``d`` form a disclosed, reproducible
calibration of that construction rather than a universal closed recurrence.
``current`` is a maintained perturbation of active returns only; the source
map is recovered at zero current.  One event denotes an active fast-cycle
return, identified from the pre-step state ``u <= u_HC``.

Reference
---------
Medvedev, G. S. (2005). *Reduction of a model of an excitable cell to a
one-dimensional map*. Physica D, 202, 37-59.
https://doi.org/10.1016/j.physd.2005.01.021
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


@dataclass(frozen=True, slots=True)
class _FirstReturnParameters:
    """Validated scalar parameters for one Medvedev first-return recurrence."""

    beta_0: float
    beta_hc: float
    beta_sn: float
    delta: float
    decay_t0: float
    alpha_t0: float
    f_0: float
    f_1: float
    homoclinic_exponent: float
    d: float
    input_gain: float

    def as_tuple(self) -> tuple[float, ...]:
        """Return parameters in the stable cross-language ABI order."""
        return (
            self.beta_0,
            self.beta_hc,
            self.beta_sn,
            self.delta,
            self.decay_t0,
            self.alpha_t0,
            self.f_0,
            self.f_1,
            self.homoclinic_exponent,
            self.d,
            self.input_gain,
        )

    @property
    def u_0(self) -> float:
        """Left/inner branch boundary induced by ``beta_0``."""
        return self.beta_0 / (self.delta - self.beta_0)

    @property
    def u_hc(self) -> float:
        """Homoclinic boundary induced by ``beta_hc``."""
        return self.beta_hc / (self.delta - self.beta_hc)

    @property
    def u_sn(self) -> float:
        """Saddle-node return state induced by ``beta_sn``."""
        return self.beta_sn / (self.delta - self.beta_sn)

    def candidate(self, u: float, current: float) -> float:
        """Evaluate the calibrated Section-4 return map without mutating state."""
        if u <= self.u_0:
            candidate = (
                self.decay_t0 * u + (1.0 - self.decay_t0) * self.f_0 + self.input_gain * current
            )
        elif u <= self.u_hc:
            u_1 = (1.0 - self.alpha_t0) * u + self.alpha_t0 * self.f_0
            gap = self.beta_hc - self.delta * u_1 / (1.0 + u_1)
            if gap <= 0.0:
                inner_return = self.f_1
            else:
                log_argument = self.d * gap
                if not math.isfinite(log_argument) or log_argument <= 0.0:
                    raise FloatingPointError("Medvedev homoclinic log argument is invalid")
                scale = math.exp(self.homoclinic_exponent * math.log(log_argument))
                inner_return = scale * (u_1 - self.f_1) + self.f_1
            candidate = inner_return + self.input_gain * current
        else:
            candidate = self.u_sn

        if not math.isfinite(candidate):
            raise FloatingPointError("Medvedev first-return candidate became non-finite")
        return candidate


class _RustSimulate(Protocol):
    """Callable surface exported by the Rust batch engine."""

    def __call__(
        self,
        u_0: float,
        beta_0: float,
        beta_hc: float,
        beta_sn: float,
        delta: float,
        decay_t0: float,
        alpha_t0: float,
        f_0: float,
        f_1: float,
        homoclinic_exponent: float,
        d: float,
        input_gain: float,
        n_steps: int,
        current: float,
    ) -> tuple[list[float], int, float]: ...


class _JuliaResult(Protocol):
    """Shape returned by the Julia ``simulate_trace`` function."""

    trace: Any
    events: int
    uf: float


class _JuliaAccel(Protocol):
    """Callable surface exposed by the loaded Julia module."""

    def simulate_trace(
        self,
        u_0: float,
        beta_0: float,
        beta_hc: float,
        beta_sn: float,
        delta: float,
        decay_t0: float,
        alpha_t0: float,
        f_0: float,
        f_1: float,
        homoclinic_exponent: float,
        d: float,
        input_gain: float,
        n_steps: int,
        current: float,
    ) -> _JuliaResult: ...


def _load_rust_simulate() -> _RustSimulate:
    engine = importlib.import_module("sc_neurocore_engine")
    return cast(_RustSimulate, engine.py_medvedev_map_simulate)


try:
    _rust_simulate: _RustSimulate | None = _load_rust_simulate()
    _HAS_RUST = True
except (ImportError, AttributeError):
    _rust_simulate = None
    _HAS_RUST = False

_julia_module: _JuliaAccel | None = None
_HAS_JULIA = False
_go_lib: ctypes.CDLL | None = None
_HAS_GO = False
_mojo_lib: ctypes.CDLL | None = None
_HAS_MOJO = False

_ACCEL_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "accel")
_AUTO_BACKENDS = with_floor("python")
_BENCHMARK_KERNEL = "medvedev_map_simulate"
_FLOAT_ARGUMENTS = 12
_MAX_C_STEPS = (1 << 31) - 1


def _ensure_julia_loaded() -> bool:
    global _julia_module, _HAS_JULIA
    if _julia_module is not None:
        return True
    if importlib.util.find_spec("juliacall") is None:
        return False
    path = os.path.abspath(os.path.join(_ACCEL_ROOT, "julia", "neurons", "medvedev_map.jl"))
    if not os.path.isfile(path):
        return False
    juliacall = importlib.import_module("juliacall")
    julia = juliacall.Main
    julia.include(path)
    _julia_module = cast(_JuliaAccel, julia.MedvedevMapAccel)
    _HAS_JULIA = True
    return True


def _load_c_backend(path: str, *, mojo: bool) -> ctypes.CDLL | None:
    if not os.path.isfile(path):
        return None
    try:
        library = ctypes.CDLL(path)
    except OSError:
        return None
    function = getattr(library, "medvedev_map_simulate_c", None)
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


def _ensure_go_loaded() -> bool:
    global _go_lib, _HAS_GO
    if _go_lib is not None:
        return True
    path = os.path.abspath(
        os.path.join(_ACCEL_ROOT, "go", "neurons", "medvedev_map", "libmedvedev.so")
    )
    _go_lib = _load_c_backend(path, mojo=False)
    _HAS_GO = _go_lib is not None
    return _HAS_GO


def _ensure_mojo_loaded() -> bool:
    global _mojo_lib, _HAS_MOJO
    if _mojo_lib is not None:
        return True
    path = os.path.abspath(os.path.join(_ACCEL_ROOT, "mojo", "neurons", "libmedvedev.so"))
    _mojo_lib = _load_c_backend(path, mojo=True)
    _HAS_MOJO = _mojo_lib is not None
    return _HAS_MOJO


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
class MedvedevMapNeuron:
    """Medvedev (2005) calibrated slow-calcium first-return map.

    Parameters
    ----------
    u : float
        Slow calcium state. The default is the calibrated saddle-node return
        ``u_SN``.
    beta_0, beta_hc, beta_sn, delta : float
        Source bifurcation parameters defining ``u_0``, ``u_HC`` and ``u_SN``.
    decay_t0, alpha_t0 : float
        Calibrated Eq. 4.4 relaxation and Eq. 4.8 affine-return coefficients.
    f_0, f_1 : float
        Calibrated fast-subsystem averages on the active and homoclinic branches.
    homoclinic_exponent, d : float
        Eq. 4.13 boundary-layer exponent and scale.
    input_gain : float
        Maintained gain for the external perturbation on active returns.
    """

    u: float = 0.2514078836724436
    beta_0: float = 0.0015
    beta_hc: float = 0.00203
    beta_sn: float = 0.002009000318382601
    delta: float = 0.01
    decay_t0: float = 0.9903563355786734
    alpha_t0: float = 0.0096904656865853
    f_0: float = 1.4713541429802286
    f_1: float = 0.1820152787145665
    homoclinic_exponent: float = 0.02149298991339221
    d: float = 2271.1927977404063
    input_gain: float = 0.01

    def __post_init__(self) -> None:
        state, parameters = self._runtime_values(construction=True)
        self.u = state
        self.beta_0 = parameters.beta_0
        self.beta_hc = parameters.beta_hc
        self.beta_sn = parameters.beta_sn
        self.delta = parameters.delta
        self.decay_t0 = parameters.decay_t0
        self.alpha_t0 = parameters.alpha_t0
        self.f_0 = parameters.f_0
        self.f_1 = parameters.f_1
        self.homoclinic_exponent = parameters.homoclinic_exponent
        self.d = parameters.d
        self.input_gain = parameters.input_gain

    def _parameters(self) -> _FirstReturnParameters:
        values = _FirstReturnParameters(
            beta_0=float(self.beta_0),
            beta_hc=float(self.beta_hc),
            beta_sn=float(self.beta_sn),
            delta=float(self.delta),
            decay_t0=float(self.decay_t0),
            alpha_t0=float(self.alpha_t0),
            f_0=float(self.f_0),
            f_1=float(self.f_1),
            homoclinic_exponent=float(self.homoclinic_exponent),
            d=float(self.d),
            input_gain=float(self.input_gain),
        )
        if not all(math.isfinite(value) for value in values.as_tuple()):
            raise ValueError("Medvedev first-return parameters must be finite")
        if not 0.0 < values.beta_0 < values.beta_sn < values.beta_hc < values.delta:
            raise ValueError("Medvedev parameters require 0 < beta_0 < beta_sn < beta_hc < delta")
        if not 0.0 < values.decay_t0 < 1.0:
            raise ValueError("decay_t0 must lie strictly between zero and one")
        if not 0.0 < values.alpha_t0 < 1.0:
            raise ValueError("alpha_t0 must lie strictly between zero and one")
        if not 0.0 <= values.f_1 < values.f_0:
            raise ValueError("calibration requires 0 <= f_1 < f_0")
        if values.homoclinic_exponent <= 0.0 or values.d <= 0.0:
            raise ValueError("homoclinic_exponent and d must be positive")
        if values.input_gain < 0.0:
            raise ValueError("input_gain must be non-negative")
        return values

    def _runtime_values(
        self, *, construction: bool = False
    ) -> tuple[float, _FirstReturnParameters]:
        state = float(self.u)
        if not math.isfinite(state):
            if construction:
                raise ValueError("Medvedev first-return state must be finite")
            raise FloatingPointError("Medvedev first-return state must be finite")
        return state, self._parameters()

    def step(self, current: float = 0.0) -> int:
        """Advance one first-return iteration without partial state mutation.

        Parameters
        ----------
        current : float, default=0.0
            Maintained perturbation applied to active returns only.

        Returns
        -------
        int
            ``1`` when the pre-state belongs to the active fast-cycle region
            ``u <= u_HC``; otherwise ``0``.

        Raises
        ------
        ValueError
            If a parameter or the current is invalid.
        FloatingPointError
            If the state or candidate is non-finite.
        """
        state, parameters = self._runtime_values()
        drive = float(current)
        if not math.isfinite(drive):
            raise ValueError("current must be finite")
        event = int(state <= parameters.u_hc)
        candidate = parameters.candidate(state, drive)
        self.u = candidate
        return event

    def simulate(
        self,
        n_steps: int,
        current: float = 0.0,
        backend: str = "auto",
    ) -> tuple[npt.NDArray[np.float64], int]:
        """Advance a constant-current first-return trajectory.

        Parameters
        ----------
        n_steps : int
            Number of map iterations, from zero through the C-ABI range.
        current : float, default=0.0
            Maintained active-return perturbation.
        backend : {"auto", "rust", "julia", "go", "mojo", "python"}
            Execution lane. ``"auto"`` follows the committed, host-matched
            benchmark order and retains Python as the guaranteed floor.

        Returns
        -------
        numpy.ndarray
            Slow-calcium value after each iteration.
        int
            Number of active fast-cycle returns.

        Raises
        ------
        ValueError
            If the request or mutable parameter set is invalid.
        RuntimeError
            If an explicitly selected compiled backend is unavailable.
        FloatingPointError
            If a backend rejects a non-finite state or candidate.
        """
        if not isinstance(n_steps, int) or isinstance(n_steps, bool):
            raise ValueError("n_steps must be an integer")
        if not 0 <= n_steps <= _MAX_C_STEPS:
            raise ValueError(f"n_steps must be between 0 and {_MAX_C_STEPS}")
        state, parameters = self._runtime_values()
        drive = float(current)
        if not math.isfinite(drive):
            raise ValueError("current must be finite")
        allowed = (*_AUTO_BACKENDS[:-1], "python", "auto")
        if backend not in allowed:
            raise ValueError(f"backend must be auto/rust/julia/go/mojo/python, got {backend!r}")
        selected = _auto_backend() if backend == "auto" else backend
        if selected != "python" and not _backend_available(selected):
            raise RuntimeError(self._unavailable_message(selected))

        if selected == "rust":
            trace, events, final_state = self._simulate_rust(n_steps, drive, state, parameters)
        elif selected == "julia":
            trace, events, final_state = self._simulate_julia(n_steps, drive, state, parameters)
        elif selected == "go":
            trace, events, final_state = self._simulate_go(n_steps, drive, state, parameters)
        elif selected == "mojo":
            trace, events, final_state = self._simulate_mojo(n_steps, drive, state, parameters)
        else:
            trace, events, final_state = self._simulate_python(n_steps, drive, state, parameters)
        self.u = final_state
        return trace, events

    @staticmethod
    def _unavailable_message(backend: str) -> str:
        if backend == "go":
            return (
                "Go Medvedev backend unavailable; build accel/go/neurons/medvedev_map/"
                "libmedvedev.so with go build -buildmode=c-shared."
            )
        if backend == "mojo":
            return (
                "Mojo Medvedev backend unavailable; build accel/mojo/neurons/"
                "libmedvedev.so with mojo build --emit shared-lib."
            )
        if backend == "julia":
            return "Julia Medvedev backend unavailable; install juliacall and Julia."
        return "Rust Medvedev backend unavailable; rebuild the sc_neurocore_engine extension."

    @staticmethod
    def _ffi_arguments(state: float, parameters: _FirstReturnParameters) -> tuple[float, ...]:
        return (
            state,
            parameters.beta_0,
            parameters.beta_hc,
            parameters.beta_sn,
            parameters.delta,
            parameters.decay_t0,
            parameters.alpha_t0,
            parameters.f_0,
            parameters.f_1,
            parameters.homoclinic_exponent,
            parameters.d,
            parameters.input_gain,
        )

    @staticmethod
    def _simulate_python(
        n_steps: int,
        current: float,
        state: float,
        parameters: _FirstReturnParameters,
    ) -> tuple[npt.NDArray[np.float64], int, float]:
        trace: npt.NDArray[np.float64] = np.empty(n_steps, dtype=np.float64)
        events = 0
        u = state
        for index in range(n_steps):
            events += int(u <= parameters.u_hc)
            u = parameters.candidate(u, current)
            trace[index] = u
        return trace, events, u

    @staticmethod
    def _simulate_rust(
        n_steps: int,
        current: float,
        state: float,
        parameters: _FirstReturnParameters,
    ) -> tuple[npt.NDArray[np.float64], int, float]:
        assert _rust_simulate is not None
        trace, events, final_state = _rust_simulate(
            state,
            parameters.beta_0,
            parameters.beta_hc,
            parameters.beta_sn,
            parameters.delta,
            parameters.decay_t0,
            parameters.alpha_t0,
            parameters.f_0,
            parameters.f_1,
            parameters.homoclinic_exponent,
            parameters.d,
            parameters.input_gain,
            n_steps,
            current,
        )
        return np.asarray(trace, dtype=np.float64), int(events), float(final_state)

    @staticmethod
    def _simulate_julia(
        n_steps: int,
        current: float,
        state: float,
        parameters: _FirstReturnParameters,
    ) -> tuple[npt.NDArray[np.float64], int, float]:
        assert _julia_module is not None
        result = _julia_module.simulate_trace(
            state,
            parameters.beta_0,
            parameters.beta_hc,
            parameters.beta_sn,
            parameters.delta,
            parameters.decay_t0,
            parameters.alpha_t0,
            parameters.f_0,
            parameters.f_1,
            parameters.homoclinic_exponent,
            parameters.d,
            parameters.input_gain,
            n_steps,
            current,
        )
        return np.asarray(result.trace, dtype=np.float64), int(result.events), float(result.uf)

    @staticmethod
    def _simulate_go(
        n_steps: int,
        current: float,
        state: float,
        parameters: _FirstReturnParameters,
    ) -> tuple[npt.NDArray[np.float64], int, float]:
        assert _go_lib is not None
        trace: npt.NDArray[np.float64] = np.zeros(n_steps + 1, dtype=np.float64, order="C")
        events = _go_lib.medvedev_map_simulate_c(
            *MedvedevMapNeuron._ffi_arguments(state, parameters),
            ctypes.c_int(n_steps),
            ctypes.c_double(current),
            trace.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        if events < 0:
            raise FloatingPointError("Go Medvedev backend rejected the recurrence")
        final_state = float(trace[n_steps]) if n_steps > 0 else state
        return np.ascontiguousarray(trace[:n_steps]), int(events), final_state

    @staticmethod
    def _simulate_mojo(
        n_steps: int,
        current: float,
        state: float,
        parameters: _FirstReturnParameters,
    ) -> tuple[npt.NDArray[np.float64], int, float]:
        assert _mojo_lib is not None
        trace: npt.NDArray[np.float64] = np.zeros(n_steps + 1, dtype=np.float64, order="C")
        events = _mojo_lib.medvedev_map_simulate_c(
            *MedvedevMapNeuron._ffi_arguments(state, parameters),
            int(n_steps),
            float(current),
            int(trace.ctypes.data),
        )
        if events < 0:
            raise FloatingPointError("Mojo Medvedev backend rejected the recurrence")
        final_state = float(trace[n_steps]) if n_steps > 0 else state
        return np.ascontiguousarray(trace[:n_steps]), int(events), final_state

    def reset(self) -> None:
        """Restore only the slow-calcium state to the calibrated ``u_SN`` return."""
        parameters = self._parameters()
        self.u = parameters.u_sn
