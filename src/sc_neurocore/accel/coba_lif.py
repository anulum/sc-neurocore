# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — COBA LIF acceleration backend loading and execution

"""Load and execute the native Brette et al. COBA LIF backends."""

from __future__ import annotations

import ctypes
import importlib
import importlib.util as importlib_util
import os
from typing import Any, Protocol, cast

import numpy as np
import numpy.typing as npt

_COBAState = tuple[float, float, float, float]
_COBAResult = tuple[npt.NDArray[np.float64], int, _COBAState]


class _EngineRunner(Protocol):
    """Typed boundary for the configurable Rust/PyO3 batch function."""

    def __call__(
        self,
        v: float,
        g_e: float,
        g_i: float,
        refractory_time: float,
        c_m: float,
        g_l: float,
        e_l: float,
        e_e: float,
        e_i: float,
        tau_e: float,
        tau_i: float,
        v_threshold: float,
        v_reset: float,
        refractory_period: float,
        dt: float,
        n_steps: int,
        current: float,
        delta_ge: float,
        delta_gi: float,
    ) -> tuple[npt.NDArray[np.float64], int, float, float, float, float]: ...


def _load_engine_coba() -> _EngineRunner:
    """Return the Rust engine's configurable COBA LIF batch function."""
    engine = importlib.import_module("sc_neurocore_engine")
    return cast(_EngineRunner, engine.py_coba_lif_simulate)


try:
    _engine_coba_simulate: _EngineRunner | None = _load_engine_coba()
    _HAS_RUST = True
except (ImportError, AttributeError):
    _engine_coba_simulate = None
    _HAS_RUST = False

_julia_module: Any | None = None
_HAS_JULIA = False
_go_lib: Any | None = None
_HAS_GO = False
_mojo_lib: Any | None = None
_HAS_MOJO = False

_ACCEL_ROOT = os.path.dirname(__file__)
_DOUBLE_FIELDS = 15


def ensure_julia_loaded() -> bool:
    """Load the executable COBA LIF Julia module when available."""
    global _julia_module, _HAS_JULIA
    if _julia_module is not None:
        return True
    if importlib_util.find_spec("juliacall") is None:
        return False
    source_path = os.path.join(_ACCEL_ROOT, "julia", "neurons", "coba_lif.jl")
    if not os.path.isfile(source_path):
        return False
    try:
        juliacall = importlib.import_module("juliacall")
        julia = juliacall.Main
        julia.include(source_path)
        _julia_module = julia.CobaLifAccel
    except (ImportError, AttributeError, RuntimeError):
        return False
    _HAS_JULIA = True
    return True


def _configure_c_library(library: Any, symbol: str, *, mojo: bool) -> Any | None:
    """Bind one COBA LIF C ABI symbol and return the configured library."""
    simulate = getattr(library, symbol, None)
    if simulate is None:
        return None
    destination = ctypes.c_int64 if mojo else ctypes.POINTER(ctypes.c_double)
    simulate.argtypes = [
        *([ctypes.c_double] * _DOUBLE_FIELDS),
        ctypes.c_int64,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        destination,
    ]
    simulate.restype = ctypes.c_int64
    return library


def ensure_go_loaded() -> bool:
    """Load the compiled COBA LIF Go C ABI when available."""
    global _go_lib, _HAS_GO
    if _go_lib is not None:
        return True
    path = os.path.join(_ACCEL_ROOT, "go", "neurons", "coba_lif", "libcoba_lif.so")
    if not os.path.isfile(path):
        return False
    try:
        library = ctypes.CDLL(path)
    except OSError:
        return False
    _go_lib = _configure_c_library(library, "coba_lif_simulate_c", mojo=False)
    _HAS_GO = _go_lib is not None
    return _HAS_GO


def ensure_mojo_loaded() -> bool:
    """Load the compiled COBA LIF Mojo C ABI when available."""
    global _mojo_lib, _HAS_MOJO
    if _mojo_lib is not None:
        return True
    path = os.path.join(_ACCEL_ROOT, "mojo", "kernels", "libcoba_lif.so")
    if not os.path.isfile(path):
        return False
    try:
        library = ctypes.CDLL(path)
    except OSError:
        return False
    _mojo_lib = _configure_c_library(library, "coba_lif_simulate_c", mojo=True)
    _HAS_MOJO = _mojo_lib is not None
    return _HAS_MOJO


def simulate_rust(
    v: float,
    g_e: float,
    g_i: float,
    refractory_time: float,
    c_m: float,
    g_l: float,
    e_l: float,
    e_e: float,
    e_i: float,
    tau_e: float,
    tau_i: float,
    v_threshold: float,
    v_reset: float,
    refractory_period: float,
    dt: float,
    n_steps: int,
    current: float,
    delta_ge: float,
    delta_gi: float,
) -> _COBAResult:
    """Run the complete contract through the production Rust engine."""
    simulate = _engine_coba_simulate
    if simulate is None:
        raise RuntimeError("Rust COBA LIF engine is unavailable.")
    result = simulate(
        v,
        g_e,
        g_i,
        refractory_time,
        c_m,
        g_l,
        e_l,
        e_e,
        e_i,
        tau_e,
        tau_i,
        v_threshold,
        v_reset,
        refractory_period,
        dt,
        n_steps,
        current,
        delta_ge,
        delta_gi,
    )
    trace, spikes, v_f, g_e_f, g_i_f, refractory_f = result
    return (
        np.ascontiguousarray(np.asarray(trace, dtype=np.float64)),
        int(spikes),
        (float(v_f), float(g_e_f), float(g_i_f), float(refractory_f)),
    )


def simulate_julia(
    v: float,
    g_e: float,
    g_i: float,
    refractory_time: float,
    c_m: float,
    g_l: float,
    e_l: float,
    e_e: float,
    e_i: float,
    tau_e: float,
    tau_i: float,
    v_threshold: float,
    v_reset: float,
    refractory_period: float,
    dt: float,
    n_steps: int,
    current: float,
    delta_ge: float,
    delta_gi: float,
) -> _COBAResult:
    """Run the Julia recurrence with the complete public contract."""
    module = _julia_module
    if module is None:
        raise RuntimeError("Julia COBA LIF module is unavailable.")
    result = module.simulate_trace(
        v,
        g_e,
        g_i,
        refractory_time,
        c_m,
        g_l,
        e_l,
        e_e,
        e_i,
        tau_e,
        tau_i,
        v_threshold,
        v_reset,
        refractory_period,
        dt,
        n_steps,
        current,
        delta_ge,
        delta_gi,
    )
    return (
        np.ascontiguousarray(np.asarray(result.trace, dtype=np.float64)),
        int(result.spikes),
        (
            float(result.v_f),
            float(result.g_e_f),
            float(result.g_i_f),
            float(result.refractory_time_f),
        ),
    )


def _simulate_c(
    library: Any,
    values: tuple[float, ...],
    n_steps: int,
    current: float,
    delta_ge: float,
    delta_gi: float,
    *,
    mojo: bool,
) -> _COBAResult:
    """Run a staged Go or Mojo COBA LIF C ABI."""
    output = np.empty(n_steps + 4, dtype=np.float64)
    destination: Any
    if mojo:
        destination = int(output.ctypes.data)
    else:
        destination = output.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    spikes = int(
        library.coba_lif_simulate_c(
            *values,
            n_steps,
            current,
            delta_ge,
            delta_gi,
            destination,
        )
    )
    if spikes < 0:
        backend = "Mojo" if mojo else "Go"
        raise FloatingPointError(f"{backend} COBA LIF kernel rejected the contract.")
    state = (
        float(output[n_steps]),
        float(output[n_steps + 1]),
        float(output[n_steps + 2]),
        float(output[n_steps + 3]),
    )
    return np.ascontiguousarray(output[:n_steps]), spikes, state


def simulate_go(
    v: float,
    g_e: float,
    g_i: float,
    refractory_time: float,
    c_m: float,
    g_l: float,
    e_l: float,
    e_e: float,
    e_i: float,
    tau_e: float,
    tau_i: float,
    v_threshold: float,
    v_reset: float,
    refractory_period: float,
    dt: float,
    n_steps: int,
    current: float,
    delta_ge: float,
    delta_gi: float,
) -> _COBAResult:
    """Run the Go recurrence through its C ABI."""
    if _go_lib is None:
        raise RuntimeError("Go COBA LIF library is unavailable.")
    values = (
        v,
        g_e,
        g_i,
        refractory_time,
        c_m,
        g_l,
        e_l,
        e_e,
        e_i,
        tau_e,
        tau_i,
        v_threshold,
        v_reset,
        refractory_period,
        dt,
    )
    return _simulate_c(_go_lib, values, n_steps, current, delta_ge, delta_gi, mojo=False)


def simulate_mojo(
    v: float,
    g_e: float,
    g_i: float,
    refractory_time: float,
    c_m: float,
    g_l: float,
    e_l: float,
    e_e: float,
    e_i: float,
    tau_e: float,
    tau_i: float,
    v_threshold: float,
    v_reset: float,
    refractory_period: float,
    dt: float,
    n_steps: int,
    current: float,
    delta_ge: float,
    delta_gi: float,
) -> _COBAResult:
    """Run the Mojo recurrence through its C ABI."""
    if _mojo_lib is None:
        raise RuntimeError("Mojo COBA LIF library is unavailable.")
    values = (
        v,
        g_e,
        g_i,
        refractory_time,
        c_m,
        g_l,
        e_l,
        e_e,
        e_i,
        tau_e,
        tau_i,
        v_threshold,
        v_reset,
        refractory_period,
        dt,
    )
    return _simulate_c(_mojo_lib, values, n_steps, current, delta_ge, delta_gi, mojo=True)
