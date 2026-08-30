# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Theta acceleration backend loading and execution

"""Load and execute the native Theta exact-flow acceleration backends."""

from __future__ import annotations

import ctypes
import importlib
import importlib.util as importlib_util
import os
from typing import Any, cast

import numpy as np
import numpy.typing as npt


def _load_engine_theta() -> tuple[type[Any], Any]:
    """Return the Rust compatibility class and checked complete batch."""
    engine = importlib.import_module("sc_neurocore_engine")
    return cast(type[Any], engine.ThetaNeuron), engine.theta_simulate_complete


_EngineThetaCls: type[Any] | None
_EngineThetaCompleteFn: Any | None
try:
    _EngineThetaCls, _EngineThetaCompleteFn = _load_engine_theta()
    _HAS_RUST = True
except (ImportError, AttributeError):
    _EngineThetaCls = None
    _EngineThetaCompleteFn = None
    _HAS_RUST = False

_julia_module: Any | None = None
_HAS_JULIA = False
_go_lib: Any | None = None
_HAS_GO = False
_mojo_lib: Any | None = None
_HAS_MOJO = False

_ACCEL_ROOT = os.path.dirname(__file__)


def ensure_julia_loaded() -> bool:
    """Load the executable Theta Julia module when available."""
    global _julia_module, _HAS_JULIA
    if _julia_module is not None:
        return True
    if importlib_util.find_spec("juliacall") is None:
        return False
    source_path = os.path.join(_ACCEL_ROOT, "julia", "neurons", "theta.jl")
    if not os.path.isfile(source_path):
        return False
    try:
        juliacall = importlib.import_module("juliacall")
        julia = juliacall.Main
        julia.include(source_path)
        _julia_module = julia.ThetaAccel
    except (ImportError, AttributeError, RuntimeError):
        return False
    _HAS_JULIA = True
    return True


def ensure_go_loaded() -> bool:
    """Load the compiled Theta Go C ABI when available."""
    global _go_lib, _HAS_GO
    if _go_lib is not None:
        return True
    library_path = os.path.join(
        _ACCEL_ROOT,
        "go",
        "neurons",
        "theta",
        "libtheta.so",
    )
    if not os.path.isfile(library_path):
        return False
    try:
        library = ctypes.CDLL(library_path)
    except OSError:
        return False
    simulate = getattr(library, "theta_simulate_c", None)
    complete = getattr(library, "theta_simulate_complete_c", None)
    if simulate is None or complete is None:
        return False
    simulate.argtypes = [
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_int64,
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_double),
    ]
    simulate.restype = ctypes.c_int64
    complete.argtypes = [
        ctypes.c_double,
        ctypes.c_double,
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
    """Load the compiled Theta Mojo C ABI when available."""
    global _mojo_lib, _HAS_MOJO
    if _mojo_lib is not None:
        return True
    library_path = os.path.join(_ACCEL_ROOT, "mojo", "kernels", "libtheta.so")
    if not os.path.isfile(library_path):
        return False
    try:
        library = ctypes.CDLL(library_path)
    except OSError:
        return False
    simulate = getattr(library, "theta_simulate_c", None)
    complete = getattr(library, "theta_simulate_complete_c", None)
    if simulate is None or complete is None:
        return False
    simulate.argtypes = [
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_int64,
        ctypes.c_double,
        ctypes.c_int64,
    ]
    simulate.restype = ctypes.c_int64
    complete.argtypes = [
        ctypes.c_double,
        ctypes.c_double,
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
    engine_class = _EngineThetaCls
    if engine_class is None:
        raise RuntimeError("Rust Theta engine is unavailable.")
    neuron = engine_class()
    trace = np.empty(n_steps, dtype=np.float64)
    spikes = 0
    for index in range(n_steps):
        spikes += int(neuron.step(float(current)))
        trace[index] = float(neuron.get_state()["theta"])
    return trace, spikes, float(neuron.get_state()["theta"])


def simulate_julia(
    theta: float,
    dt: float,
    n_steps: int,
    current: float,
) -> tuple[npt.NDArray[np.float64], int, float]:
    """Run the Julia recurrence with the complete numeric contract."""
    module = _julia_module
    if module is None:
        raise RuntimeError("Julia Theta module is unavailable.")
    result = module.simulate_trace(float(theta), float(dt), int(n_steps), float(current))
    trace = np.ascontiguousarray(np.asarray(result.trace, dtype=np.float64))
    return trace, int(result.spikes), float(result.thetaf)


def simulate_go(
    theta: float,
    dt: float,
    n_steps: int,
    current: float,
) -> tuple[npt.NDArray[np.float64], int, float]:
    """Run the Go recurrence through its C ABI."""
    library = _go_lib
    if library is None:
        raise RuntimeError("Go Theta library is unavailable.")
    output = np.empty(n_steps + 1, dtype=np.float64)
    spikes = int(
        library.theta_simulate_c(
            theta,
            dt,
            n_steps,
            current,
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
    )
    if spikes < 0:
        raise FloatingPointError("Go Theta kernel rejected the simulation contract.")
    return np.ascontiguousarray(output[:n_steps]), spikes, float(output[n_steps])


def simulate_mojo(
    theta: float,
    dt: float,
    n_steps: int,
    current: float,
) -> tuple[npt.NDArray[np.float64], int, float]:
    """Run the Mojo recurrence through its C ABI."""
    library = _mojo_lib
    if library is None:
        raise RuntimeError("Mojo Theta library is unavailable.")
    output = np.empty(n_steps + 1, dtype=np.float64)
    spikes = int(
        library.theta_simulate_c(
            theta,
            dt,
            n_steps,
            current,
            int(output.ctypes.data),
        )
    )
    if spikes < 0:
        raise FloatingPointError("Mojo Theta kernel rejected the simulation contract.")
    return np.ascontiguousarray(output[:n_steps]), spikes, float(output[n_steps])


def simulate_rust_complete(
    theta: float, dt: float, n_steps: int, current: float
) -> tuple[object, object, float]:
    """Run the checked phase-explicit production Rust batch."""
    if _EngineThetaCompleteFn is None:
        raise RuntimeError("Rust Theta complete batch is unavailable.")
    result = _EngineThetaCompleteFn(theta, dt, n_steps, current)
    return result[0], result[1], float(result[2])


def simulate_julia_complete(
    theta: float, dt: float, n_steps: int, current: float
) -> tuple[object, object, float]:
    """Run the checked phase-explicit Julia batch."""
    if _julia_module is None:
        raise RuntimeError("Julia Theta module is unavailable.")
    result = _julia_module.simulate_complete(theta, dt, n_steps, current)
    return result.trace, result.events, float(result.thetaf)


def simulate_go_complete(
    theta: float, dt: float, n_steps: int, current: float
) -> tuple[object, object, float]:
    """Run the failure-atomic Go complete C-ABI packet."""
    if _go_lib is None:
        raise RuntimeError("Go Theta library is unavailable.")
    phase = np.empty(n_steps + 1, dtype=np.float64)
    events = np.empty(n_steps, dtype=np.uint8)
    count = int(
        _go_lib.theta_simulate_complete_c(
            theta,
            dt,
            n_steps,
            current,
            phase.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            events.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        )
    )
    if count < 0 or count != int(np.sum(events, dtype=np.int64)):
        raise FloatingPointError("Go Theta kernel rejected its packet.")
    return np.ascontiguousarray(phase[:n_steps]), events, float(phase[n_steps])


def simulate_mojo_complete(
    theta: float, dt: float, n_steps: int, current: float
) -> tuple[object, object, float]:
    """Run the failure-atomic Mojo complete C-ABI packet."""
    if _mojo_lib is None:
        raise RuntimeError("Mojo Theta library is unavailable.")
    phase = np.empty(n_steps + 1, dtype=np.float64)
    events = np.empty(n_steps, dtype=np.uint8)
    count = int(
        _mojo_lib.theta_simulate_complete_c(
            theta,
            dt,
            n_steps,
            current,
            int(phase.ctypes.data),
            int(events.ctypes.data),
        )
    )
    if count < 0 or count != int(np.sum(events, dtype=np.int64)):
        raise FloatingPointError("Mojo Theta kernel rejected its packet.")
    return np.ascontiguousarray(phase[:n_steps]), events, float(phase[n_steps])
