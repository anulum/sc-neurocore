# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — retained upward-crossing Rulkov-map identity

"""Retained SC-NeuroCore upward-crossing Rulkov-map recurrence.

This count-neutral identity preserves the behavior historically exposed as
``RulkovMapNeuron``: the Rulkov (2002) state recurrence combined with a
configurable upward crossing of ``x_threshold``. The source-faithful
``RulkovMapNeuron`` instead reports execution of the paper's rightmost reset
branch. Keeping the identities separate prevents a scientific correction from
silently deleting an established project behavior.
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
from sc_neurocore.neurons.models.rulkov_map import (
    RulkovMapNeuron,
    _MAX_C_STEPS,
    _RulkovParameters,
)


class _RustSimulate(Protocol):
    """Callable surface exported by the retained Rust batch engine."""

    def __call__(
        self,
        x_0: float,
        y_0: float,
        alpha: float,
        sigma: float,
        mu: float,
        x_threshold: float,
        n_steps: int,
        current: float,
    ) -> tuple[Any, int, float, float]: ...


class _JuliaResult(Protocol):
    """Shape returned by the retained Julia batch function."""

    trace: Any
    events: int
    xf: float
    yf: float


class _JuliaAccel(Protocol):
    """Callable surface exposed by the retained Julia module."""

    def simulate_trace(
        self,
        x_0: float,
        y_0: float,
        alpha: float,
        sigma: float,
        mu: float,
        x_threshold: float,
        n_steps: int,
        current: float,
    ) -> _JuliaResult: ...


def _load_rust_simulate() -> _RustSimulate:
    engine = importlib.import_module("sc_neurocore_engine")
    return cast(_RustSimulate, engine.py_sc_upward_crossing_rulkov_map_simulate)


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
_BENCHMARK_KERNEL = "sc_upward_crossing_rulkov_map_simulate"
_FLOAT_ARGUMENTS = 6


def _ensure_julia_loaded() -> bool:
    global _julia_module, _HAS_JULIA
    if _julia_module is not None:
        return True
    if importlib.util.find_spec("juliacall") is None:
        return False
    path = os.path.abspath(
        os.path.join(_ACCEL_ROOT, "julia", "neurons", "sc_upward_crossing_rulkov_map.jl")
    )
    if not os.path.isfile(path):
        return False
    juliacall = importlib.import_module("juliacall")
    julia = juliacall.Main
    julia.include(path)
    _julia_module = cast(_JuliaAccel, julia.SCUpwardCrossingRulkovMapAccel)
    _HAS_JULIA = True
    return True


def _load_c_backend(path: str, *, mojo: bool) -> ctypes.CDLL | None:
    if not os.path.isfile(path):
        return None
    try:
        library = ctypes.CDLL(path)
    except OSError:
        return None
    function = getattr(library, "sc_upward_crossing_rulkov_map_simulate_c", None)
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
        os.path.join(
            _ACCEL_ROOT,
            "go",
            "neurons",
            "sc_upward_crossing_rulkov_map",
            "libsc_upward_crossing_rulkov_map.so",
        )
    )
    _go_lib = _load_c_backend(path, mojo=False)
    _HAS_GO = _go_lib is not None
    return _HAS_GO


def _ensure_mojo_loaded() -> bool:
    global _mojo_lib, _HAS_MOJO
    if _mojo_lib is not None:
        return True
    path = os.path.abspath(
        os.path.join(_ACCEL_ROOT, "mojo", "neurons", "libsc_upward_crossing_rulkov_map.so")
    )
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
class SCUpwardCrossingRulkovMapNeuron(RulkovMapNeuron):
    """Retained configurable upward-crossing observation convention."""

    x_threshold: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        self.x_threshold = float(self.x_threshold)
        if not math.isfinite(self.x_threshold):
            raise ValueError("x_threshold must be finite")

    def _validated_threshold(self) -> float:
        threshold = float(self.x_threshold)
        if not math.isfinite(threshold):
            raise ValueError("x_threshold must be finite")
        return threshold

    def step(self, current: float = 0.0) -> int:
        """Advance once and report the historical rising threshold crossing."""
        drive = float(current)
        if not math.isfinite(drive):
            raise ValueError("current must be finite")
        x, y = self._validated_state()
        threshold = self._validated_threshold()
        x_next, y_next, _source_event = self._parameters().candidate(x, y, drive)
        event = int(x_next >= threshold and x < threshold)
        self.x, self.y = x_next, y_next
        return event

    def simulate(
        self, n_steps: int, current: float = 0.0, backend: str = "auto"
    ) -> tuple[npt.NDArray[np.float64], int]:
        """Advance a failure-atomic retained batch and return trace plus events."""
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
        threshold = self._validated_threshold()
        self._validated_state()

        selected = _auto_backend() if backend == "auto" else backend
        if selected != "python" and not _backend_available(selected):
            raise RuntimeError(f"{selected} SC upward-crossing Rulkov backend is unavailable")
        if selected == "rust":
            result = self._simulate_sc_rust(n_steps, drive, parameters, threshold)
        elif selected == "julia":
            result = self._simulate_sc_julia(n_steps, drive, parameters, threshold)
        elif selected == "go":
            result = self._simulate_sc_go(n_steps, drive, parameters, threshold)
        elif selected == "mojo":
            result = self._simulate_sc_mojo(n_steps, drive, parameters, threshold)
        else:
            result = self._simulate_sc_python(n_steps, drive, parameters, threshold)
        trace, events, x_final, y_final = result
        self.x, self.y = x_final, y_final
        return trace, events

    def _simulate_sc_python(
        self,
        n_steps: int,
        current: float,
        parameters: _RulkovParameters,
        threshold: float,
    ) -> tuple[npt.NDArray[np.float64], int, float, float]:
        trace = np.empty(n_steps, dtype=np.float64)
        x, y = self.x, self.y
        events = 0
        for index in range(n_steps):
            x_previous = x
            x, y, _source_event = parameters.candidate(x, y, current)
            trace[index] = x
            events += int(x >= threshold and x_previous < threshold)
        return trace, events, x, y

    def _simulate_sc_rust(
        self,
        n_steps: int,
        current: float,
        parameters: _RulkovParameters,
        threshold: float,
    ) -> tuple[npt.NDArray[np.float64], int, float, float]:
        if _rust_simulate is None:
            raise RuntimeError("rust SC upward-crossing Rulkov backend is unavailable")
        trace, events, x_final, y_final = _rust_simulate(
            self.x, self.y, *parameters.as_tuple(), threshold, n_steps, current
        )
        return np.asarray(trace, dtype=np.float64), int(events), float(x_final), float(y_final)

    def _simulate_sc_julia(
        self,
        n_steps: int,
        current: float,
        parameters: _RulkovParameters,
        threshold: float,
    ) -> tuple[npt.NDArray[np.float64], int, float, float]:
        if _julia_module is None:
            raise RuntimeError("julia SC upward-crossing Rulkov backend is unavailable")
        try:
            result = _julia_module.simulate_trace(
                self.x, self.y, *parameters.as_tuple(), threshold, n_steps, current
            )
        except Exception as error:
            if (
                error.__class__.__module__ != "juliacall"
                or error.__class__.__name__ != "JuliaError"
            ):
                raise
            raise FloatingPointError(
                "Julia retained Rulkov backend rejected an invalid candidate"
            ) from error
        return (
            np.asarray(result.trace, dtype=np.float64),
            int(result.events),
            float(result.xf),
            float(result.yf),
        )

    def _simulate_sc_go(
        self,
        n_steps: int,
        current: float,
        parameters: _RulkovParameters,
        threshold: float,
    ) -> tuple[npt.NDArray[np.float64], int, float, float]:
        if _go_lib is None:
            raise RuntimeError("go SC upward-crossing Rulkov backend is unavailable")
        trace = np.zeros(n_steps + 2, dtype=np.float64, order="C")
        events = _go_lib.sc_upward_crossing_rulkov_map_simulate_c(
            ctypes.c_double(self.x),
            ctypes.c_double(self.y),
            *(ctypes.c_double(value) for value in parameters.as_tuple()),
            ctypes.c_double(threshold),
            ctypes.c_int(n_steps),
            ctypes.c_double(current),
            trace.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        if events < 0:
            raise FloatingPointError("Go retained Rulkov backend rejected an invalid candidate")
        return (
            np.ascontiguousarray(trace[:n_steps]),
            int(events),
            trace[n_steps],
            trace[n_steps + 1],
        )

    def _simulate_sc_mojo(
        self,
        n_steps: int,
        current: float,
        parameters: _RulkovParameters,
        threshold: float,
    ) -> tuple[npt.NDArray[np.float64], int, float, float]:
        if _mojo_lib is None:
            raise RuntimeError("mojo SC upward-crossing Rulkov backend is unavailable")
        trace = np.zeros(n_steps + 2, dtype=np.float64, order="C")
        events = _mojo_lib.sc_upward_crossing_rulkov_map_simulate_c(
            self.x,
            self.y,
            *parameters.as_tuple(),
            threshold,
            n_steps,
            current,
            int(trace.ctypes.data),
        )
        if events < 0:
            raise FloatingPointError("Mojo retained Rulkov backend rejected an invalid candidate")
        return (
            np.ascontiguousarray(trace[:n_steps]),
            int(events),
            trace[n_steps],
            trace[n_steps + 1],
        )


__all__ = ["SCUpwardCrossingRulkovMapNeuron"]
