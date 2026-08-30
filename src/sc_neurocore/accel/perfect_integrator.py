# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Perfect-integrator acceleration backend execution

"""Load and execute profile-explicit native perfect-integrator backends."""

from __future__ import annotations

import ctypes
import importlib
import importlib.util as importlib_util
import os
from typing import Any, cast

import numpy as np


def _load_engine_perfect_integrator() -> tuple[type[Any], Any]:
    """Return the Rust compatibility class and complete-batch entrypoint."""
    engine = importlib.import_module("sc_neurocore_engine")
    return (
        cast(type[Any], engine.PerfectIntegratorNeuron),
        engine.perfect_integrator_simulate_complete,
    )


_EnginePerfectIntegratorCls: type[Any] | None
_EnginePerfectIntegratorCompleteFn: Any | None
try:
    (
        _EnginePerfectIntegratorCls,
        _EnginePerfectIntegratorCompleteFn,
    ) = _load_engine_perfect_integrator()
    _HAS_RUST = True
except (ImportError, AttributeError):
    _EnginePerfectIntegratorCls = None
    _EnginePerfectIntegratorCompleteFn = None
    _HAS_RUST = False

_julia_module: Any | None = None
_HAS_JULIA = False
_go_lib: Any | None = None
_HAS_GO = False
_mojo_lib: Any | None = None
_HAS_MOJO = False
_ACCEL_ROOT = os.path.dirname(__file__)


def ensure_julia_loaded() -> bool:
    """Load the executable perfect-integrator Julia module when available."""
    global _julia_module, _HAS_JULIA
    if _julia_module is not None:
        return True
    if importlib_util.find_spec("juliacall") is None:
        return False
    source_path = os.path.join(_ACCEL_ROOT, "julia", "neurons", "perfect_integrator.jl")
    if not os.path.isfile(source_path):
        return False
    try:
        juliacall = importlib.import_module("juliacall")
        julia = juliacall.Main
        julia.include(source_path)
        _julia_module = julia.PerfectIntegratorAccel
    except (ImportError, AttributeError, RuntimeError):
        return False
    _HAS_JULIA = True
    return True


def ensure_go_loaded() -> bool:
    """Load the compiled perfect-integrator Go C ABI when available."""
    global _go_lib, _HAS_GO
    if _go_lib is not None:
        return True
    library_path = os.path.join(
        _ACCEL_ROOT, "go", "neurons", "perfect_integrator", "libperfect_integrator.so"
    )
    if not os.path.isfile(library_path):
        return False
    try:
        library = ctypes.CDLL(library_path)
    except OSError:
        return False
    legacy = getattr(library, "perfect_integrator_simulate_c", None)
    complete = getattr(library, "perfect_integrator_simulate_complete_c", None)
    if legacy is None or complete is None:
        return False
    legacy.argtypes = [ctypes.c_double] * 5 + [
        ctypes.c_int64,
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_double),
    ]
    legacy.restype = ctypes.c_int64
    complete.argtypes = [ctypes.c_double] * 5 + [
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
    """Load the compiled perfect-integrator Mojo C ABI when available."""
    global _mojo_lib, _HAS_MOJO
    if _mojo_lib is not None:
        return True
    library_path = os.path.join(_ACCEL_ROOT, "mojo", "kernels", "libperfect_integrator.so")
    if not os.path.isfile(library_path):
        return False
    try:
        library = ctypes.CDLL(library_path)
    except OSError:
        return False
    legacy = getattr(library, "perfect_integrator_simulate_c", None)
    complete = getattr(library, "perfect_integrator_simulate_complete_c", None)
    if legacy is None or complete is None:
        return False
    legacy.argtypes = [ctypes.c_double] * 5 + [
        ctypes.c_int64,
        ctypes.c_double,
        ctypes.c_int64,
    ]
    legacy.restype = ctypes.c_int64
    complete.argtypes = [ctypes.c_double] * 5 + [
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


def simulate_rust_complete(
    v: float,
    c_m: float,
    v_threshold: float,
    v_reset: float,
    dt: float,
    source_profile: bool,
    n_steps: int,
    current: float,
) -> tuple[object, object, float]:
    """Run the complete profile-explicit Rust production batch."""
    assert _EnginePerfectIntegratorCompleteFn is not None
    result = _EnginePerfectIntegratorCompleteFn(
        v, c_m, v_threshold, v_reset, dt, source_profile, n_steps, current
    )
    return result[0], result[1], float(result[2])


def simulate_julia_complete(
    v: float,
    c_m: float,
    v_threshold: float,
    v_reset: float,
    dt: float,
    source_profile: bool,
    n_steps: int,
    current: float,
) -> tuple[object, object, float]:
    """Run the complete profile-explicit Julia batch."""
    assert _julia_module is not None
    result = _julia_module.simulate_complete(
        v, c_m, v_threshold, v_reset, dt, source_profile, n_steps, current
    )
    return result.trace, result.events, float(result.vf)


def simulate_go_complete(
    v: float,
    c_m: float,
    v_threshold: float,
    v_reset: float,
    dt: float,
    source_profile: bool,
    n_steps: int,
    current: float,
) -> tuple[object, object, float]:
    """Run the complete Go batch through its mutation-free C ABI."""
    assert _go_lib is not None
    voltage = np.empty(n_steps + 1, dtype=np.float64)
    events = np.empty(n_steps, dtype=np.uint8)
    count = int(
        _go_lib.perfect_integrator_simulate_complete_c(
            v,
            c_m,
            v_threshold,
            v_reset,
            dt,
            int(source_profile),
            n_steps,
            current,
            voltage.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            events.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        )
    )
    if count < 0 or count != int(np.sum(events, dtype=np.int64)):
        raise FloatingPointError("Go PerfectIntegrator kernel rejected its packet.")
    return np.ascontiguousarray(voltage[:n_steps]), events, float(voltage[n_steps])


def simulate_mojo_complete(
    v: float,
    c_m: float,
    v_threshold: float,
    v_reset: float,
    dt: float,
    source_profile: bool,
    n_steps: int,
    current: float,
) -> tuple[object, object, float]:
    """Run the complete Mojo batch through its mutation-free C ABI."""
    assert _mojo_lib is not None
    voltage = np.empty(n_steps + 1, dtype=np.float64)
    events = np.empty(n_steps, dtype=np.uint8)
    count = int(
        _mojo_lib.perfect_integrator_simulate_complete_c(
            v,
            c_m,
            v_threshold,
            v_reset,
            dt,
            int(source_profile),
            n_steps,
            current,
            int(voltage.ctypes.data),
            int(events.ctypes.data),
        )
    )
    if count < 0 or count != int(np.sum(events, dtype=np.int64)):
        raise FloatingPointError("Mojo PerfectIntegrator kernel rejected its packet.")
    return np.ascontiguousarray(voltage[:n_steps]), events, float(voltage[n_steps])
