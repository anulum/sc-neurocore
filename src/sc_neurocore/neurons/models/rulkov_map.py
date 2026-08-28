# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Rulkov 2002 piecewise fast/slow map

r"""Rulkov (2002) two-dimensional fast/slow map neuron.

The source model is a simultaneous discrete recurrence. With the repository's
scalar input bound to the paper's fast ``beta_n`` channel, it is

.. math::

   x_{n+1} =
   \begin{cases}
   \alpha/(1-x_n) + y_n + I_n, & x_n \le 0,\\
   \alpha + y_n + I_n, & 0 < x_n < \alpha+y_n+I_n,\\
   -1, & x_n \ge \alpha+y_n+I_n,
   \end{cases}

.. math::

   y_{n+1} = y_n - \mu(x_n+1) + \mu\sigma.

Following the source's event convention, a spike is the iteration whose
pre-update fast state occupies the rightmost branch. The former configurable
upward-``x``-crossing convention remains available, without a literature-model
count, as :class:`SCUpwardCrossingRulkovMapNeuron`.

Reference
---------
Rulkov, N. F. (2002). *Modeling of spiking-bursting neural behavior using
two-dimensional map*. Physical Review E, 65, 041922.
https://doi.org/10.1103/PhysRevE.65.041922
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
class _RulkovParameters:
    """Validated parameters for the Rulkov (2002) recurrence."""

    alpha: float
    sigma: float
    mu: float

    def as_tuple(self) -> tuple[float, float, float]:
        """Return parameters in the stable cross-language ABI order."""
        return self.alpha, self.sigma, self.mu

    def candidate(self, x: float, y: float, current: float) -> tuple[float, float, int]:
        """Evaluate one simultaneous source-map iteration without mutation."""
        boundary = self.alpha + y + current
        if not math.isfinite(boundary):
            raise FloatingPointError("Rulkov map branch boundary became non-finite")
        reset_event = x > 0.0 and x >= boundary
        if x <= 0.0:
            denominator = 1.0 - x
            if denominator <= 0.0 or not math.isfinite(denominator):
                raise FloatingPointError("Rulkov map branch denominator is invalid")
            x_next = self.alpha / denominator + y + current
        elif x < boundary:
            x_next = boundary
        else:
            x_next = -1.0
        y_next = y - self.mu * (x + 1.0) + self.mu * self.sigma
        if not math.isfinite(x_next) or not math.isfinite(y_next):
            raise FloatingPointError("Rulkov map candidate state became non-finite")
        return x_next, y_next, int(reset_event)


class _RustSimulate(Protocol):
    """Callable surface exported by the Rust batch engine."""

    def __call__(
        self,
        x_0: float,
        y_0: float,
        alpha: float,
        sigma: float,
        mu: float,
        n_steps: int,
        current: float,
    ) -> tuple[Any, int, float, float]: ...


class _JuliaResult(Protocol):
    """Shape returned by the Julia batch function."""

    trace: Any
    events: int
    xf: float
    yf: float


class _JuliaAccel(Protocol):
    """Callable surface exposed by the loaded Julia module."""

    def simulate_trace(
        self,
        x_0: float,
        y_0: float,
        alpha: float,
        sigma: float,
        mu: float,
        n_steps: int,
        current: float,
    ) -> _JuliaResult: ...


def _load_rust_simulate() -> _RustSimulate:
    engine = importlib.import_module("sc_neurocore_engine")
    return cast(_RustSimulate, engine.py_rulkov_map_simulate)


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
_BENCHMARK_KERNEL = "rulkov_map_simulate"
_FLOAT_ARGUMENTS = 5
_MAX_C_STEPS = (1 << 31) - 1


def _ensure_julia_loaded() -> bool:
    global _julia_module, _HAS_JULIA
    if _julia_module is not None:
        return True
    if importlib.util.find_spec("juliacall") is None:
        return False
    path = os.path.abspath(os.path.join(_ACCEL_ROOT, "julia", "neurons", "rulkov_map.jl"))
    if not os.path.isfile(path):
        return False
    juliacall = importlib.import_module("juliacall")
    julia = juliacall.Main
    julia.include(path)
    _julia_module = cast(_JuliaAccel, julia.RulkovMapAccel)
    _HAS_JULIA = True
    return True


def _load_c_backend(path: str, *, mojo: bool) -> ctypes.CDLL | None:
    if not os.path.isfile(path):
        return None
    try:
        library = ctypes.CDLL(path)
    except OSError:
        return None
    function = getattr(library, "rulkov_map_simulate_c", None)
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
    path = os.path.abspath(os.path.join(_ACCEL_ROOT, "go", "neurons", "rulkov_map", "librulkov.so"))
    _go_lib = _load_c_backend(path, mojo=False)
    _HAS_GO = _go_lib is not None
    return _HAS_GO


def _ensure_mojo_loaded() -> bool:
    global _mojo_lib, _HAS_MOJO
    if _mojo_lib is not None:
        return True
    path = os.path.abspath(os.path.join(_ACCEL_ROOT, "mojo", "neurons", "librulkov.so"))
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
class RulkovMapNeuron:
    """Rulkov (2002) three-branch spiking-bursting map.

    ``current`` is the paper's fast ``beta_n`` input specialization. ``sigma``
    remains the slow-nullcline control parameter of the autonomous map. The
    default state and ``sigma=-1.6`` form SC-NeuroCore's quiescent operating
    profile; they are not represented as a source figure's unique defaults.
    """

    x: float = -1.0
    y: float = -3.0
    alpha: float = 4.0
    sigma: float = -1.6
    mu: float = 0.001

    def __post_init__(self) -> None:
        self.x = float(self.x)
        self.y = float(self.y)
        self.alpha = float(self.alpha)
        self.sigma = float(self.sigma)
        self.mu = float(self.mu)
        if not math.isfinite(self.x) or not math.isfinite(self.y):
            raise ValueError("Rulkov map state must be finite")
        self._parameters()

    def _validated_state(self) -> tuple[float, float]:
        x = float(self.x)
        y = float(self.y)
        if not math.isfinite(x) or not math.isfinite(y):
            raise FloatingPointError("Rulkov map state must be finite")
        return x, y

    def _parameters(self) -> _RulkovParameters:
        alpha = float(self.alpha)
        sigma = float(self.sigma)
        mu = float(self.mu)
        if not all(math.isfinite(value) for value in (alpha, sigma, mu)):
            raise ValueError("Rulkov map parameters must be finite")
        if alpha <= 0.0:
            raise ValueError("alpha must be positive")
        if mu <= 0.0:
            raise ValueError("mu must be positive")
        return _RulkovParameters(alpha, sigma, mu)

    def step(self, current: float = 0.0) -> int:
        """Advance one source-map iteration and report the reset-branch event."""
        drive = float(current)
        if not math.isfinite(drive):
            raise ValueError("current must be finite")
        x, y = self._validated_state()
        x_next, y_next, event = self._parameters().candidate(x, y, drive)
        self.x, self.y = x_next, y_next
        return event

    def simulate(
        self, n_steps: int, current: float = 0.0, backend: str = "auto"
    ) -> tuple[npt.NDArray[np.float64], int]:
        """Advance a failure-atomic batch and return ``(x_trace, events)``."""
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
        parameters = self._parameters()
        self._validated_state()

        selected = _auto_backend() if backend == "auto" else backend
        if selected != "python" and not _backend_available(selected):
            raise RuntimeError(f"{selected} Rulkov backend is unavailable")
        if selected == "rust":
            trace, events, x_final, y_final = self._simulate_rust(n_steps, drive, parameters)
        elif selected == "julia":
            trace, events, x_final, y_final = self._simulate_julia(n_steps, drive, parameters)
        elif selected == "go":
            trace, events, x_final, y_final = self._simulate_go(n_steps, drive, parameters)
        elif selected == "mojo":
            trace, events, x_final, y_final = self._simulate_mojo(n_steps, drive, parameters)
        else:
            trace, events, x_final, y_final = self._simulate_python(n_steps, drive, parameters)
        self.x, self.y = x_final, y_final
        return trace, events

    def _simulate_python(
        self, n_steps: int, current: float, parameters: _RulkovParameters
    ) -> tuple[npt.NDArray[np.float64], int, float, float]:
        trace = np.empty(n_steps, dtype=np.float64)
        x, y = self.x, self.y
        events = 0
        for index in range(n_steps):
            x, y, event = parameters.candidate(x, y, current)
            trace[index] = x
            events += event
        return trace, events, x, y

    def _simulate_rust(
        self, n_steps: int, current: float, parameters: _RulkovParameters
    ) -> tuple[npt.NDArray[np.float64], int, float, float]:
        if _rust_simulate is None:
            raise RuntimeError("rust Rulkov backend is unavailable")
        trace, events, x_final, y_final = _rust_simulate(
            self.x, self.y, *parameters.as_tuple(), n_steps, current
        )
        return np.asarray(trace, dtype=np.float64), int(events), float(x_final), float(y_final)

    def _simulate_julia(
        self, n_steps: int, current: float, parameters: _RulkovParameters
    ) -> tuple[npt.NDArray[np.float64], int, float, float]:
        if _julia_module is None:
            raise RuntimeError("julia Rulkov backend is unavailable")
        try:
            result = _julia_module.simulate_trace(
                self.x, self.y, *parameters.as_tuple(), n_steps, current
            )
        except Exception as error:
            if (
                error.__class__.__module__ != "juliacall"
                or error.__class__.__name__ != "JuliaError"
            ):
                raise
            raise FloatingPointError(
                "Julia Rulkov backend rejected an invalid candidate"
            ) from error
        return (
            np.asarray(result.trace, dtype=np.float64),
            int(result.events),
            float(result.xf),
            float(result.yf),
        )

    def _simulate_go(
        self, n_steps: int, current: float, parameters: _RulkovParameters
    ) -> tuple[npt.NDArray[np.float64], int, float, float]:
        if _go_lib is None:
            raise RuntimeError("go Rulkov backend is unavailable")
        trace = np.zeros(n_steps + 2, dtype=np.float64, order="C")
        events = _go_lib.rulkov_map_simulate_c(
            ctypes.c_double(self.x),
            ctypes.c_double(self.y),
            *(ctypes.c_double(value) for value in parameters.as_tuple()),
            ctypes.c_int(n_steps),
            ctypes.c_double(current),
            trace.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        if events < 0:
            raise FloatingPointError("Go Rulkov backend rejected an invalid candidate")
        return (
            np.ascontiguousarray(trace[:n_steps]),
            int(events),
            trace[n_steps],
            trace[n_steps + 1],
        )

    def _simulate_mojo(
        self, n_steps: int, current: float, parameters: _RulkovParameters
    ) -> tuple[npt.NDArray[np.float64], int, float, float]:
        if _mojo_lib is None:
            raise RuntimeError("mojo Rulkov backend is unavailable")
        trace = np.zeros(n_steps + 2, dtype=np.float64, order="C")
        events = _mojo_lib.rulkov_map_simulate_c(
            self.x,
            self.y,
            *parameters.as_tuple(),
            n_steps,
            current,
            int(trace.ctypes.data),
        )
        if events < 0:
            raise FloatingPointError("Mojo Rulkov backend rejected an invalid candidate")
        return (
            np.ascontiguousarray(trace[:n_steps]),
            int(events),
            trace[n_steps],
            trace[n_steps + 1],
        )

    def reset(self) -> None:
        """Restore the repository operating profile's initial state."""
        self.x = -1.0
        self.y = -3.0


__all__ = ["RulkovMapNeuron"]
