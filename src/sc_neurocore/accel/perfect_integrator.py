# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Perfect Integrator acceleration backend loading and execution

"""Load and execute the native Perfect Integrator acceleration backends."""

from __future__ import annotations

import ctypes
import importlib
import importlib.util as importlib_util
import os
from typing import Any, cast

import numpy as np
import numpy.typing as npt


def _load_engine_perfect_integrator() -> type[Any]:
    """Return the Rust engine's factory-default Perfect Integrator class."""
    engine = importlib.import_module("sc_neurocore_engine")
    return cast(type[Any], engine.PerfectIntegratorNeuron)


try:
    _EnginePerfectIntegratorCls: type[Any] | None = _load_engine_perfect_integrator()
    _HAS_RUST = True
except (ImportError, AttributeError):
    _EnginePerfectIntegratorCls = None
    _HAS_RUST = False

_julia_module: Any | None = None
_HAS_JULIA = False
_go_lib: Any | None = None
_HAS_GO = False
_mojo_lib: Any | None = None
_HAS_MOJO = False

_ACCEL_ROOT = os.path.dirname(__file__)


def ensure_julia_loaded() -> bool:
    """Load the executable Perfect Integrator Julia module when available."""
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
    """Load the compiled Perfect Integrator Go C ABI when available."""
    global _go_lib, _HAS_GO
    if _go_lib is not None:
        return True
    library_path = os.path.join(
        _ACCEL_ROOT,
        "go",
        "neurons",
        "perfect_integrator",
        "libperfect_integrator.so",
    )
    if not os.path.isfile(library_path):
        return False
    try:
        library = ctypes.CDLL(library_path)
    except OSError:
        return False
    simulate = getattr(library, "perfect_integrator_simulate_c", None)
    if simulate is None:
        return False
    simulate.argtypes = [ctypes.c_double] * 5 + [
        ctypes.c_int64,
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_double),
    ]
    simulate.restype = ctypes.c_int64
    _go_lib = library
    _HAS_GO = True
    return True


def ensure_mojo_loaded() -> bool:
    """Load the compiled Perfect Integrator Mojo C ABI when available."""
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
    simulate = getattr(library, "perfect_integrator_simulate_c", None)
    if simulate is None:
        return False
    simulate.argtypes = [ctypes.c_double] * 5 + [
        ctypes.c_int64,
        ctypes.c_double,
        ctypes.c_int64,
    ]
    simulate.restype = ctypes.c_int64
    _mojo_lib = library
    _HAS_MOJO = True
    return True


def simulate_rust(n_steps: int, current: float) -> tuple[npt.NDArray[np.float64], int, float]:
    """Run the factory-default Rust engine recurrence."""
    assert _EnginePerfectIntegratorCls is not None
    neuron = _EnginePerfectIntegratorCls()
    trace = np.empty(n_steps, dtype=np.float64)
    spikes = 0
    for index in range(n_steps):
        spikes += int(neuron.step(float(current)))
        trace[index] = float(neuron.get_state()["v"])
    return trace, spikes, float(neuron.get_state()["v"])


def simulate_julia(
    v: float,
    c_m: float,
    v_threshold: float,
    v_reset: float,
    dt: float,
    n_steps: int,
    current: float,
) -> tuple[npt.NDArray[np.float64], int, float]:
    """Run the Julia recurrence with the complete numeric contract."""
    assert _julia_module is not None
    result = _julia_module.simulate_trace(
        float(v),
        float(c_m),
        float(v_threshold),
        float(v_reset),
        float(dt),
        int(n_steps),
        float(current),
    )
    trace = np.ascontiguousarray(np.asarray(result.trace, dtype=np.float64))
    return trace, int(result.spikes), float(result.vf)


def simulate_go(
    v: float,
    c_m: float,
    v_threshold: float,
    v_reset: float,
    dt: float,
    n_steps: int,
    current: float,
) -> tuple[npt.NDArray[np.float64], int, float]:
    """Run the Go recurrence through its C ABI."""
    assert _go_lib is not None
    output = np.empty(n_steps + 1, dtype=np.float64)
    spikes = int(
        _go_lib.perfect_integrator_simulate_c(
            v,
            c_m,
            v_threshold,
            v_reset,
            dt,
            n_steps,
            current,
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
    )
    if spikes < 0:
        raise FloatingPointError("Go Perfect Integrator kernel rejected the simulation contract.")
    return np.ascontiguousarray(output[:n_steps]), spikes, float(output[n_steps])


def simulate_mojo(
    v: float,
    c_m: float,
    v_threshold: float,
    v_reset: float,
    dt: float,
    n_steps: int,
    current: float,
) -> tuple[npt.NDArray[np.float64], int, float]:
    """Run the Mojo recurrence through its C ABI."""
    assert _mojo_lib is not None
    output = np.empty(n_steps + 1, dtype=np.float64)
    spikes = int(
        _mojo_lib.perfect_integrator_simulate_c(
            v,
            c_m,
            v_threshold,
            v_reset,
            dt,
            n_steps,
            current,
            int(output.ctypes.data),
        )
    )
    if spikes < 0:
        raise FloatingPointError("Mojo Perfect Integrator kernel rejected the simulation contract.")
    return np.ascontiguousarray(output[:n_steps]), spikes, float(output[n_steps])
