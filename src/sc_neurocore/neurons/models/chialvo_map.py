# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Chialvo 1995 two-dimensional discrete map neuron

"""Source-faithful Chialvo two-dimensional discrete map.

Chialvo's recurrence is evaluated simultaneously once per call to
:meth:`ChialvoMapNeuron.step`:

``x[n+1] = x[n]**2 * exp(y[n] - x[n]) + k + I[n]``

``y[n+1] = a * y[n] - b * x[n] + c``

The source permits ``k`` to be a constant bias or a time-dependent additive
perturbation. SC-NeuroCore represents the constant part as ``k`` and supplies
the time-dependent part through ``current``. The upward ``x_threshold`` event
is a maintained observation convention; it is not attributed to the paper.

Reference
---------
Chialvo, D. R. (1995). *Generic excitable dynamics on a two-dimensional map*.
Chaos, Solitons & Fractals, 5(3-4), 461-479.
https://doi.org/10.1016/0960-0779(93)E0056-H
"""

from __future__ import annotations

import ctypes
import importlib
import importlib.util
import math
import os
from dataclasses import dataclass
from typing import Any, Callable, Protocol, cast

import numpy as np
import numpy.typing as npt

from sc_neurocore.accel.backend_order import with_floor
from sc_neurocore.accel.backend_selection import select_backend_order
from sc_neurocore.utils.numerics import safe_exp

_RustSimulate = Callable[
    [float, float, float, float, float, float, float, int, float],
    tuple[list[float], int, float, float],
]
_AUTO_BACKENDS = with_floor("python")
_BENCHMARK_KERNEL = "chialvo_map_simulate"


class _JuliaResult(Protocol):
    """Shape returned by the Julia ``simulate_trace`` function."""

    trace: Any
    spikes: int
    xf: float
    yf: float


class _JuliaAccel(Protocol):
    """Callable surface exposed by the loaded Julia module."""

    def simulate_trace(
        self,
        x0: float,
        y0: float,
        a: float,
        b: float,
        c: float,
        k: float,
        x_threshold: float,
        n_steps: int,
        current: float,
    ) -> _JuliaResult: ...


def _load_rust_simulate() -> _RustSimulate:
    engine = importlib.import_module("sc_neurocore_engine")
    return cast(_RustSimulate, engine.py_chialvo_map_simulate)


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


def _ensure_julia_loaded() -> bool:
    global _julia_module, _HAS_JULIA
    if _julia_module is not None:
        return True
    if importlib.util.find_spec("juliacall") is None:
        return False
    jl_path = os.path.abspath(os.path.join(_ACCEL_ROOT, "julia", "neurons", "chialvo_map.jl"))
    if not os.path.isfile(jl_path):
        return False
    juliacall = importlib.import_module("juliacall")
    jl = juliacall.Main
    jl.include(jl_path)
    _julia_module = cast(_JuliaAccel, jl.ChialvoMapAccel)
    _HAS_JULIA = True
    return True


def _load_c_backend(path: str, symbol: str, *, mojo: bool) -> ctypes.CDLL | None:
    if not os.path.isfile(path):
        return None
    try:
        lib = ctypes.CDLL(path)
    except OSError:
        return None
    function = getattr(lib, symbol, None)
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
    return lib


def _ensure_go_loaded() -> bool:
    global _go_lib, _HAS_GO
    if _go_lib is not None:
        return True
    path = os.path.abspath(
        os.path.join(_ACCEL_ROOT, "go", "neurons", "chialvo_map", "libchialvo.so")
    )
    _go_lib = _load_c_backend(path, "chialvo_map_simulate_c", mojo=False)
    _HAS_GO = _go_lib is not None
    return _HAS_GO


def _ensure_mojo_loaded() -> bool:
    global _mojo_lib, _HAS_MOJO
    if _mojo_lib is not None:
        return True
    path = os.path.abspath(os.path.join(_ACCEL_ROOT, "mojo", "neurons", "libchialvo.so"))
    _mojo_lib = _load_c_backend(path, "chialvo_map_simulate_c", mojo=True)
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
    ordered: tuple[str, ...] = select_backend_order(_BENCHMARK_KERNEL, static=_AUTO_BACKENDS)
    for backend in ordered:
        if _backend_available(backend):
            return backend
    return "python"


@dataclass
class ChialvoMapNeuron:
    """Chialvo (1995) two-dimensional discrete map neuron.

    Parameters
    ----------
    x, y : float
        Fast and recovery state variables.
    a, b, c, k : float
        Published dimensionless map parameters. The defaults reproduce a
        parameter set used in the source paper.
    x_threshold : float
        Maintained upward-crossing observation level; not a source parameter.
    """

    x: float = 0.0
    y: float = 0.0
    a: float = 0.89
    b: float = 0.6
    c: float = 0.28
    k: float = 0.04
    x_threshold: float = 1.0

    def __post_init__(self) -> None:
        self._validate_configuration()

    def _validate_configuration(self) -> None:
        for name in ("x", "y", "a", "b", "c", "k", "x_threshold"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            setattr(self, name, value)

    def _validate_runtime(self) -> None:
        x = float(self.x)
        y = float(self.y)
        if not math.isfinite(x) or not math.isfinite(y):
            raise FloatingPointError("Chialvo map runtime state must be finite")
        self.x, self.y = x, y
        for name in ("a", "b", "c", "k", "x_threshold"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            setattr(self, name, value)

    def _candidate(self, x: float, y: float, current: float) -> tuple[float, float]:
        x_squared = x * x
        exponential = safe_exp(y - x)
        x_next = x_squared * exponential + self.k + current
        y_next = self.a * y - self.b * x + self.c
        if not math.isfinite(x_next) or not math.isfinite(y_next):
            raise FloatingPointError("Chialvo map candidate state became non-finite")
        return x_next, y_next

    def step(self, current: float = 0.0) -> int:
        """Advance one simultaneous map iteration.

        Parameters
        ----------
        current : float, default=0.0
            Additive perturbation applied to the fast-variable recurrence.

        Returns
        -------
        int
            ``1`` on an upward crossing of ``x_threshold``; otherwise ``0``.

        Raises
        ------
        ValueError
            If the configuration or current is non-finite.
        FloatingPointError
            If the candidate state cannot be represented as finite ``float64``.
        """
        self._validate_runtime()
        drive = float(current)
        if not math.isfinite(drive):
            raise ValueError("current must be finite")
        x_previous = self.x
        x_next, y_next = self._candidate(self.x, self.y, drive)
        self.x, self.y = x_next, y_next
        return int(x_previous < self.x_threshold <= x_next)

    def simulate(
        self,
        n_steps: int,
        current: float = 0.0,
        backend: str = "auto",
    ) -> tuple[npt.NDArray[np.float64], int]:
        """Advance several map iterations through a selected backend.

        Parameters
        ----------
        n_steps : int
            Number of discrete map iterations; must be non-negative.
        current : float, default=0.0
            Constant additive perturbation for every iteration.
        backend : {"auto", "rust", "julia", "go", "mojo", "python"}
            Execution lane. ``"auto"`` uses the committed host-matched
            benchmark order, then falls back through available lanes to Python.

        Returns
        -------
        numpy.ndarray
            Fast-variable value after each iteration.
        int
            Number of maintained upward-threshold events.

        Raises
        ------
        ValueError
            If an argument or mutable configuration is invalid.
        RuntimeError
            If an explicitly requested compiled backend is unavailable.
        FloatingPointError
            If a backend rejects a non-finite candidate state.
        """
        if n_steps < 0:
            raise ValueError("n_steps must be non-negative")
        self._validate_runtime()
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
            trace, spikes, x_final, y_final = self._simulate_rust(n_steps, drive)
        elif selected == "julia":
            trace, spikes, x_final, y_final = self._simulate_julia(n_steps, drive)
        elif selected == "go":
            trace, spikes, x_final, y_final = self._simulate_go(n_steps, drive)
        elif selected == "mojo":
            trace, spikes, x_final, y_final = self._simulate_mojo(n_steps, drive)
        else:
            trace, spikes, x_final, y_final = self._simulate_python(n_steps, drive)
        self.x, self.y = x_final, y_final
        return trace, spikes

    @staticmethod
    def _unavailable_message(backend: str) -> str:
        if backend == "go":
            return (
                "Go Chialvo backend unavailable; build accel/go/neurons/chialvo_map/"
                "libchialvo.so with go build -buildmode=c-shared."
            )
        if backend == "mojo":
            return (
                "Mojo Chialvo backend unavailable; build accel/mojo/neurons/"
                "libchialvo.so from chialvo_map.mojo."
            )
        return f"{backend.title()} Chialvo backend is unavailable"

    def _simulate_python(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, float, float]:
        trace: npt.NDArray[np.float64] = np.empty(n_steps, dtype=np.float64)
        x, y = self.x, self.y
        spikes = 0
        for index in range(n_steps):
            x_previous = x
            x, y = self._candidate(x, y, current)
            trace[index] = x
            spikes += int(x_previous < self.x_threshold <= x)
        return trace, spikes, x, y

    def _simulate_rust(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, float, float]:
        if _rust_simulate is None:
            raise RuntimeError(self._unavailable_message("rust"))
        trace, spikes, x_final, y_final = _rust_simulate(
            self.x,
            self.y,
            self.a,
            self.b,
            self.c,
            self.k,
            self.x_threshold,
            n_steps,
            current,
        )
        return np.asarray(trace, dtype=np.float64), int(spikes), float(x_final), float(y_final)

    def _simulate_julia(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, float, float]:
        if _julia_module is None:
            raise RuntimeError(self._unavailable_message("julia"))
        result = _julia_module.simulate_trace(
            self.x,
            self.y,
            self.a,
            self.b,
            self.c,
            self.k,
            self.x_threshold,
            n_steps,
            current,
        )
        return (
            np.asarray(result.trace, dtype=np.float64),
            int(result.spikes),
            float(result.xf),
            float(result.yf),
        )

    def _simulate_go(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, float, float]:
        if _go_lib is None:
            raise RuntimeError(self._unavailable_message("go"))
        return self._simulate_c(_go_lib, n_steps, current, mojo=False)

    def _simulate_mojo(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, float, float]:
        if _mojo_lib is None:
            raise RuntimeError(self._unavailable_message("mojo"))
        return self._simulate_c(_mojo_lib, n_steps, current, mojo=True)

    def _simulate_c(
        self,
        library: ctypes.CDLL,
        n_steps: int,
        current: float,
        *,
        mojo: bool,
    ) -> tuple[npt.NDArray[np.float64], int, float, float]:
        trace: npt.NDArray[np.float64] = np.empty(n_steps + 2, dtype=np.float64)
        args: list[Any] = [
            self.x,
            self.y,
            self.a,
            self.b,
            self.c,
            self.k,
            self.x_threshold,
            n_steps,
            current,
        ]
        if mojo:
            args.append(int(trace.ctypes.data))
        else:
            args.append(trace.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        spikes = int(library.chialvo_map_simulate_c(*args))
        if spikes < 0:
            raise FloatingPointError("Chialvo compiled backend rejected the candidate state")
        return (
            np.ascontiguousarray(trace[:n_steps]),
            spikes,
            float(trace[n_steps]),
            float(trace[n_steps + 1]),
        )

    def reset(self) -> None:
        """Restore the two state variables while preserving configured parameters."""
        self.x, self.y = 0.0, 0.0
