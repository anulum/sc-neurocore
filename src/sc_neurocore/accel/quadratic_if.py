# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Quadratic IF acceleration backend loading and execution

"""Load and execute the native Quadratic IF acceleration backends."""

from __future__ import annotations

import ctypes
import importlib
import importlib.util as importlib_util
import os
from typing import Any, cast

import numpy as np
import numpy.typing as npt


def _load_engine_quadratic_if() -> tuple[type[Any], Any]:
    """Return the Rust compatibility class and checked complete batch."""
    engine = importlib.import_module("sc_neurocore_engine")
    return cast(type[Any], engine.QuadraticIFNeuron), engine.quadratic_if_simulate_complete


_EngineQuadraticIFCls: type[Any] | None
_EngineQuadraticIFCompleteFn: Any | None
try:
    _EngineQuadraticIFCls, _EngineQuadraticIFCompleteFn = _load_engine_quadratic_if()
    _HAS_RUST = True
except (ImportError, AttributeError):
    _EngineQuadraticIFCls = None
    _EngineQuadraticIFCompleteFn = None
    _HAS_RUST = False

_julia_module: Any | None = None
_HAS_JULIA = False
_go_lib: Any | None = None
_HAS_GO = False
_mojo_lib: Any | None = None
_HAS_MOJO = False

_ACCEL_ROOT = os.path.dirname(__file__)


def ensure_julia_loaded() -> bool:
    """Load the executable Quadratic IF Julia module when available."""
    global _julia_module, _HAS_JULIA
    if _julia_module is not None:
        return True
    if importlib_util.find_spec("juliacall") is None:
        return False
    source_path = os.path.join(_ACCEL_ROOT, "julia", "neurons", "quadratic_if.jl")
    if not os.path.isfile(source_path):
        return False
    try:
        juliacall = importlib.import_module("juliacall")
        julia = juliacall.Main
        julia.include(source_path)
        _julia_module = julia.QuadraticIfAccel
    except (ImportError, AttributeError, RuntimeError):
        return False
    _HAS_JULIA = True
    return True


def ensure_go_loaded() -> bool:
    """Load the compiled Quadratic IF Go C ABI when available."""
    global _go_lib, _HAS_GO
    if _go_lib is not None:
        return True
    library_path = os.path.join(
        _ACCEL_ROOT,
        "go",
        "neurons",
        "quadratic_if",
        "libquadratic_if.so",
    )
    if not os.path.isfile(library_path):
        return False
    try:
        library = ctypes.CDLL(library_path)
    except OSError:
        return False
    simulate = getattr(library, "quadratic_if_simulate_c", None)
    complete = getattr(library, "quadratic_if_simulate_complete_c", None)
    if simulate is None or complete is None:
        return False
    simulate.argtypes = [ctypes.c_double] * 4 + [
        ctypes.c_int64,
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_double),
    ]
    simulate.restype = ctypes.c_int64
    complete.argtypes = [ctypes.c_double] * 4 + [
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_uint8),
    ]
    complete.restype = ctypes.c_int64
    _go_lib = library
    _HAS_GO = True
    return True


def ensure_mojo_loaded() -> bool:
    """Load the compiled Quadratic IF Mojo C ABI when available."""
    global _mojo_lib, _HAS_MOJO
    if _mojo_lib is not None:
        return True
    library_path = os.path.join(_ACCEL_ROOT, "mojo", "kernels", "libquadratic_if.so")
    if not os.path.isfile(library_path):
        return False
    try:
        library = ctypes.CDLL(library_path)
    except OSError:
        return False
    simulate = getattr(library, "quadratic_if_simulate_c", None)
    complete = getattr(library, "quadratic_if_simulate_complete_c", None)
    if simulate is None or complete is None:
        return False
    simulate.argtypes = [ctypes.c_double] * 4 + [
        ctypes.c_int64,
        ctypes.c_double,
        ctypes.c_int64,
    ]
    simulate.restype = ctypes.c_int64
    complete.argtypes = [ctypes.c_double] * 4 + [
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_double,
        ctypes.c_int64,
        ctypes.c_int64,
    ]
    complete.restype = ctypes.c_int64
    _mojo_lib = library
    _HAS_MOJO = True
    return True


def simulate_rust(n_steps: int, current: float) -> tuple[npt.NDArray[np.float64], int, float]:
    """Run the factory-default Rust engine exact-flow recurrence."""
    engine_class = _EngineQuadraticIFCls
    if engine_class is None:
        raise RuntimeError("Rust Quadratic IF engine is unavailable.")
    neuron = engine_class()
    trace = np.empty(n_steps, dtype=np.float64)
    spikes = 0
    for index in range(n_steps):
        spikes += int(neuron.step(float(current)))
        trace[index] = float(neuron.get_state()["v"])
    return trace, spikes, float(neuron.get_state()["v"])


def simulate_julia(
    v: float,
    v_reset: float,
    v_peak: float,
    dt: float,
    n_steps: int,
    current: float,
) -> tuple[npt.NDArray[np.float64], int, float]:
    """Run the Julia recurrence with the complete numeric contract."""
    module = _julia_module
    if module is None:
        raise RuntimeError("Julia Quadratic IF module is unavailable.")
    result = module.simulate_trace(
        float(v),
        float(v_reset),
        float(v_peak),
        float(dt),
        int(n_steps),
        float(current),
    )
    trace = np.ascontiguousarray(np.asarray(result.trace, dtype=np.float64))
    return trace, int(result.spikes), float(result.vf)


def simulate_go(
    v: float,
    v_reset: float,
    v_peak: float,
    dt: float,
    n_steps: int,
    current: float,
) -> tuple[npt.NDArray[np.float64], int, float]:
    """Run the Go recurrence through its C ABI."""
    library = _go_lib
    if library is None:
        raise RuntimeError("Go Quadratic IF library is unavailable.")
    output = np.empty(n_steps + 1, dtype=np.float64)
    spikes = int(
        library.quadratic_if_simulate_c(
            v,
            v_reset,
            v_peak,
            dt,
            n_steps,
            current,
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
    )
    if spikes < 0:
        raise FloatingPointError("Go Quadratic IF kernel rejected the simulation contract.")
    return np.ascontiguousarray(output[:n_steps]), spikes, float(output[n_steps])


def simulate_mojo(
    v: float,
    v_reset: float,
    v_peak: float,
    dt: float,
    n_steps: int,
    current: float,
) -> tuple[npt.NDArray[np.float64], int, float]:
    """Run the Mojo recurrence through its C ABI."""
    library = _mojo_lib
    if library is None:
        raise RuntimeError("Mojo Quadratic IF library is unavailable.")
    output = np.empty(n_steps + 1, dtype=np.float64)
    spikes = int(
        library.quadratic_if_simulate_c(
            v,
            v_reset,
            v_peak,
            dt,
            n_steps,
            current,
            int(output.ctypes.data),
        )
    )
    if spikes < 0:
        raise FloatingPointError("Mojo Quadratic IF kernel rejected the simulation contract.")
    return np.ascontiguousarray(output[:n_steps]), spikes, float(output[n_steps])


def simulate_rust_complete(
    v: float,
    v_reset: float,
    v_peak: float,
    dt: float,
    source_profile: bool,
    n_steps: int,
    current: float,
) -> tuple[object, object, float]:
    """Run the checked profile-explicit production Rust batch."""
    if _EngineQuadraticIFCompleteFn is None:
        raise RuntimeError("Rust Quadratic IF complete batch is unavailable.")
    result = _EngineQuadraticIFCompleteFn(v, v_reset, v_peak, dt, source_profile, n_steps, current)
    return result[0], result[1], float(result[2])


def simulate_julia_complete(
    v: float,
    v_reset: float,
    v_peak: float,
    dt: float,
    source_profile: bool,
    n_steps: int,
    current: float,
) -> tuple[object, object, float]:
    """Run the checked profile-explicit Julia batch."""
    if _julia_module is None:
        raise RuntimeError("Julia Quadratic IF module is unavailable.")
    result = _julia_module.simulate_complete(
        v, v_reset, v_peak, dt, source_profile, n_steps, current
    )
    return result.trace, result.events, float(result.vf)


def simulate_go_complete(
    v: float,
    v_reset: float,
    v_peak: float,
    dt: float,
    source_profile: bool,
    n_steps: int,
    current: float,
) -> tuple[object, object, float]:
    """Run the failure-atomic Go complete C-ABI packet."""
    if _go_lib is None:
        raise RuntimeError("Go Quadratic IF library is unavailable.")
    voltage = np.empty(n_steps + 1, dtype=np.float64)
    events = np.empty(n_steps, dtype=np.uint8)
    count = int(
        _go_lib.quadratic_if_simulate_complete_c(
            v,
            v_reset,
            v_peak,
            dt,
            int(source_profile),
            n_steps,
            current,
            voltage.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            events.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        )
    )
    if count < 0 or count != int(np.sum(events, dtype=np.int64)):
        raise FloatingPointError("Go Quadratic IF kernel rejected its packet.")
    return np.ascontiguousarray(voltage[:n_steps]), events, float(voltage[n_steps])


def simulate_mojo_complete(
    v: float,
    v_reset: float,
    v_peak: float,
    dt: float,
    source_profile: bool,
    n_steps: int,
    current: float,
) -> tuple[object, object, float]:
    """Run the failure-atomic Mojo complete C-ABI packet."""
    if _mojo_lib is None:
        raise RuntimeError("Mojo Quadratic IF library is unavailable.")
    voltage = np.empty(n_steps + 1, dtype=np.float64)
    events = np.empty(n_steps, dtype=np.uint8)
    count = int(
        _mojo_lib.quadratic_if_simulate_complete_c(
            v,
            v_reset,
            v_peak,
            dt,
            int(source_profile),
            n_steps,
            current,
            int(voltage.ctypes.data),
            int(events.ctypes.data),
        )
    )
    if count < 0 or count != int(np.sum(events, dtype=np.int64)):
        raise FloatingPointError("Mojo Quadratic IF kernel rejected its packet.")
    return np.ascontiguousarray(voltage[:n_steps]), events, float(voltage[n_steps])
