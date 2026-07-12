# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Ibarz-Tanaka 2007 four-branch Rulkov map

r"""Source-derived Ibarz-Tanaka spiking-bursting map.

Ibarz, Tanaka, Sanjuan and Aihara analyse the self-sustained-oscillation
variant of the Rulkov map. Their Eqs. 2-3 define a simultaneous two-state
recurrence. For ``h = I + u`` the fast state is

.. math::

   v_{n+1} =
   \begin{cases}
   -\alpha^2/4 - \alpha + h, & v < -1 - \alpha/2,\\
   \alpha v + (v + 1)^2 + h, & -1 - \alpha/2 \le v \le 0,\\
   1 + h, & 0 < v < 1 + h,\\
   -1, & v \ge 1 + h,
   \end{cases}

and ``u[n+1] = u[n] - mu * (v[n] + 1 - sigma)``. An event is recorded when
the fourth branch performs the source reset. The external ``current`` is the
paper's ``I_v`` input. No configurable ``beta``, fixed threshold, or separate
reset parameter belongs to this model.

Reference
---------
Ibarz, B., Tanaka, G., Sanjuan, M. A. F. & Aihara, K. (2007). *Sensitivity
versus resonance in two-dimensional spiking-bursting neuron models*.
Physical Review E, 75, 041902. https://doi.org/10.1103/PhysRevE.75.041902
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
class _MapParameters:
    """Validated parameters for the four-branch source recurrence."""

    alpha: float
    mu: float
    sigma: float

    def as_tuple(self) -> tuple[float, float, float]:
        """Return parameters in the stable cross-language ABI order."""
        return self.alpha, self.mu, self.sigma

    def candidate(self, v: float, u: float, current: float) -> tuple[float, float, int]:
        """Evaluate Eqs. 2-3 without mutating the caller's state."""
        lower = -1.0 - self.alpha / 2.0
        upper = 1.0 + current + u
        if v < lower:
            v_next = -(self.alpha * self.alpha) / 4.0 - self.alpha + current + u
        elif v <= 0.0:
            v_next = self.alpha * v + (v + 1.0) * (v + 1.0) + current + u
        elif v < upper:
            v_next = upper
        else:
            v_next = -1.0
        u_next = u - self.mu * (v + 1.0 - self.sigma)
        if not math.isfinite(v_next) or not math.isfinite(u_next):
            raise FloatingPointError("Ibarz-Tanaka map candidate became non-finite")
        return v_next, u_next, int(v >= upper)


class _RustSimulate(Protocol):
    """Callable surface exported by the Rust batch engine."""

    def __call__(
        self,
        v_0: float,
        u_0: float,
        alpha: float,
        mu: float,
        sigma: float,
        n_steps: int,
        current: float,
    ) -> tuple[list[float], int, float, float]: ...


class _JuliaResult(Protocol):
    """Shape returned by the Julia batch function."""

    trace: Any
    events: int
    vf: float
    uf: float


class _JuliaAccel(Protocol):
    """Callable surface exposed by the loaded Julia module."""

    def simulate_trace(
        self,
        v_0: float,
        u_0: float,
        alpha: float,
        mu: float,
        sigma: float,
        n_steps: int,
        current: float,
    ) -> _JuliaResult: ...


def _load_rust_simulate() -> _RustSimulate:
    engine = importlib.import_module("sc_neurocore_engine")
    return cast(_RustSimulate, engine.py_ibarz_tanaka_map_simulate)


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
_BENCHMARK_KERNEL = "ibarz_tanaka_map_simulate"
_FLOAT_ARGUMENTS = 5
_MAX_C_STEPS = (1 << 31) - 1


def _ensure_julia_loaded() -> bool:
    global _julia_module, _HAS_JULIA
    if _julia_module is not None:
        return True
    if importlib.util.find_spec("juliacall") is None:
        return False
    path = os.path.abspath(os.path.join(_ACCEL_ROOT, "julia", "neurons", "ibarz_tanaka_map.jl"))
    if not os.path.isfile(path):
        return False
    juliacall = importlib.import_module("juliacall")
    julia = juliacall.Main
    julia.include(path)
    _julia_module = cast(_JuliaAccel, julia.IbarzTanakaMapAccel)
    _HAS_JULIA = True
    return True


def _load_c_backend(path: str, *, mojo: bool) -> ctypes.CDLL | None:
    if not os.path.isfile(path):
        return None
    try:
        library = ctypes.CDLL(path)
    except OSError:
        return None
    function = getattr(library, "ibarz_tanaka_map_simulate_c", None)
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
        os.path.join(_ACCEL_ROOT, "go", "neurons", "ibarz_tanaka_map", "libibarz.so")
    )
    _go_lib = _load_c_backend(path, mojo=False)
    _HAS_GO = _go_lib is not None
    return _HAS_GO


def _ensure_mojo_loaded() -> bool:
    global _mojo_lib, _HAS_MOJO
    if _mojo_lib is not None:
        return True
    path = os.path.abspath(os.path.join(_ACCEL_ROOT, "mojo", "neurons", "libibarz.so"))
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
class IbarzTanakaMapNeuron:
    """Ibarz et al. (2007) four-branch Rulkov map.

    Parameters
    ----------
    v, u : float
        Fast membrane-like state and slow recovery state. The defaults reproduce
        the map placement used for Fig. 2(a) of the source at zero current.
    alpha : float
        Fast-map geometry parameter from Eq. 3.
    mu : float
        Positive slow timescale from Eq. 2.
    sigma : float
        Slow-nullcline offset from Eq. 2.
    """

    v: float = -1.0
    u: float = -0.1
    alpha: float = 1.0
    mu: float = 0.001
    sigma: float = 0.1

    def __post_init__(self) -> None:
        self._parameters()
        if not math.isfinite(self.v) or not math.isfinite(self.u):
            raise ValueError("Ibarz-Tanaka state must be finite")

    def _parameters(self) -> _MapParameters:
        values = (self.alpha, self.mu, self.sigma)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Ibarz-Tanaka parameters must be finite")
        if self.alpha <= 0.0:
            raise ValueError("alpha must be positive")
        if self.mu <= 0.0:
            raise ValueError("mu must be positive")
        return _MapParameters(*values)

    def step(self, current: float) -> int:
        """Advance Eqs. 2-3 once and return the reset-branch event."""
        if not math.isfinite(float(current)):
            raise ValueError("current must be finite")
        v_next, u_next, event = self._parameters().candidate(self.v, self.u, current)
        self.v, self.u = v_next, u_next
        return event

    def simulate(
        self, n_steps: int, current: float = 0.0, backend: str = "auto"
    ) -> tuple[npt.NDArray[np.float64], int]:
        """Advance the map and return the fast-state trace plus event count."""
        if isinstance(n_steps, bool) or not isinstance(n_steps, int):
            raise ValueError("n_steps must be an integer")
        if not 0 <= n_steps <= _MAX_C_STEPS:
            raise ValueError(f"n_steps must be between 0 and {_MAX_C_STEPS}")
        if not math.isfinite(float(current)):
            raise ValueError("current must be finite")
        if backend not in {"auto", "rust", "julia", "go", "mojo", "python"}:
            raise ValueError(f"unsupported backend: {backend}")
        parameters = self._parameters()
        if not math.isfinite(self.v) or not math.isfinite(self.u):
            raise ValueError("Ibarz-Tanaka state must be finite")

        selected = _auto_backend() if backend == "auto" else backend
        if selected != "python" and not _backend_available(selected):
            raise RuntimeError(f"{selected} Ibarz-Tanaka backend is unavailable")
        if selected == "rust":
            trace, events, v_final, u_final = self._simulate_rust(n_steps, current, parameters)
        elif selected == "julia":
            trace, events, v_final, u_final = self._simulate_julia(n_steps, current, parameters)
        elif selected == "go":
            trace, events, v_final, u_final = self._simulate_go(n_steps, current, parameters)
        elif selected == "mojo":
            trace, events, v_final, u_final = self._simulate_mojo(n_steps, current, parameters)
        else:
            trace, events, v_final, u_final = self._simulate_python(n_steps, current, parameters)
        self.v, self.u = v_final, u_final
        return trace, events

    def _simulate_python(
        self, n_steps: int, current: float, parameters: _MapParameters
    ) -> tuple[npt.NDArray[np.float64], int, float, float]:
        trace = np.empty(n_steps, dtype=np.float64)
        v, u = self.v, self.u
        events = 0
        for index in range(n_steps):
            v, u, event = parameters.candidate(v, u, current)
            trace[index] = v
            events += event
        return trace, events, v, u

    def _simulate_rust(
        self, n_steps: int, current: float, parameters: _MapParameters
    ) -> tuple[npt.NDArray[np.float64], int, float, float]:
        assert _rust_simulate is not None
        trace, events, v_final, u_final = _rust_simulate(
            self.v, self.u, *parameters.as_tuple(), n_steps, current
        )
        return np.asarray(trace, dtype=np.float64), int(events), float(v_final), float(u_final)

    def _simulate_julia(
        self, n_steps: int, current: float, parameters: _MapParameters
    ) -> tuple[npt.NDArray[np.float64], int, float, float]:
        assert _julia_module is not None
        result = _julia_module.simulate_trace(
            self.v, self.u, *parameters.as_tuple(), n_steps, current
        )
        return (
            np.asarray(result.trace, dtype=np.float64),
            int(result.events),
            float(result.vf),
            float(result.uf),
        )

    def _simulate_go(
        self, n_steps: int, current: float, parameters: _MapParameters
    ) -> tuple[npt.NDArray[np.float64], int, float, float]:
        assert _go_lib is not None
        trace = np.zeros(n_steps + 2, dtype=np.float64, order="C")
        events = _go_lib.ibarz_tanaka_map_simulate_c(
            ctypes.c_double(self.v),
            ctypes.c_double(self.u),
            *(ctypes.c_double(value) for value in parameters.as_tuple()),
            ctypes.c_int(n_steps),
            ctypes.c_double(current),
            trace.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        return (
            np.ascontiguousarray(trace[:n_steps]),
            int(events),
            trace[n_steps],
            trace[n_steps + 1],
        )

    def _simulate_mojo(
        self, n_steps: int, current: float, parameters: _MapParameters
    ) -> tuple[npt.NDArray[np.float64], int, float, float]:
        assert _mojo_lib is not None
        trace = np.zeros(n_steps + 2, dtype=np.float64, order="C")
        events = _mojo_lib.ibarz_tanaka_map_simulate_c(
            self.v,
            self.u,
            *parameters.as_tuple(),
            n_steps,
            current,
            int(trace.ctypes.data),
        )
        return (
            np.ascontiguousarray(trace[:n_steps]),
            int(events),
            trace[n_steps],
            trace[n_steps + 1],
        )

    def reset(self) -> None:
        """Restore the source example's initial state without changing parameters."""
        self.v = -1.0
        self.u = -0.1
