# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — DPI acceleration backend loading and execution

"""Load and execute the native current-mode DPI neuron backends."""

from __future__ import annotations

import ctypes
import importlib
import importlib.util as importlib_util
import os
from typing import Any, cast

import numpy as np
import numpy.typing as npt

_DPIState = tuple[float, float, float]
_DPIResult = tuple[npt.NDArray[np.float64], int, _DPIState]
_DPICompleteResult = tuple[object, object, object, object, _DPIState]


def _load_engine_dpi() -> tuple[type[Any], Any]:
    """Return the Rust compatibility class and complete batch function."""
    engine = importlib.import_module("sc_neurocore_engine")
    return cast(type[Any], engine.DPINeuron), engine.dpi_neuron_simulate_complete


_EngineDPICls: type[Any] | None
_EngineDPICompleteFn: Any | None
try:
    _EngineDPICls, _EngineDPICompleteFn = _load_engine_dpi()
    _HAS_RUST = True
except (ImportError, AttributeError):
    _EngineDPICls = None
    _EngineDPICompleteFn = None
    _HAS_RUST = False

_julia_module: Any | None = None
_HAS_JULIA = False
_go_lib: Any | None = None
_HAS_GO = False
_mojo_lib: Any | None = None
_HAS_MOJO = False

_ACCEL_ROOT = os.path.dirname(__file__)
_DOUBLE_FIELDS = 18


def ensure_julia_loaded() -> bool:
    """Load the executable DPI Julia module when available."""
    global _julia_module, _HAS_JULIA
    if _julia_module is not None:
        return True
    if importlib_util.find_spec("juliacall") is None:
        return False
    source_path = os.path.join(_ACCEL_ROOT, "julia", "neurons", "dpi_neuron.jl")
    if not os.path.isfile(source_path):
        return False
    try:
        juliacall = importlib.import_module("juliacall")
        julia = juliacall.Main
        julia.include(source_path)
        _julia_module = julia.DpiNeuronAccel
    except (ImportError, AttributeError, RuntimeError):
        return False
    _HAS_JULIA = True
    return True


def _configure_c_library(library: Any, symbol: str, *, mojo: bool) -> Any | None:
    """Bind the compatibility and complete DPI C ABI symbols."""
    simulate = getattr(library, symbol, None)
    complete = getattr(library, "dpi_neuron_simulate_complete_c", None)
    if simulate is None or complete is None:
        return None
    tail = ctypes.c_int64 if mojo else ctypes.POINTER(ctypes.c_double)
    simulate.argtypes = [
        *([ctypes.c_double] * _DOUBLE_FIELDS),
        ctypes.c_int64,
        ctypes.c_double,
        tail,
    ]
    simulate.restype = ctypes.c_int64
    complete_tail = ctypes.c_int64 if mojo else ctypes.POINTER(ctypes.c_double)
    event_tail = ctypes.c_int64 if mojo else ctypes.POINTER(ctypes.c_uint8)
    complete.argtypes = [
        *([ctypes.c_double] * _DOUBLE_FIELDS),
        ctypes.c_int64,
        ctypes.c_double,
        complete_tail,
        complete_tail,
        complete_tail,
        event_tail,
    ]
    complete.restype = ctypes.c_int64
    return library


def ensure_go_loaded() -> bool:
    """Load the compiled DPI Go C ABI when available."""
    global _go_lib, _HAS_GO
    if _go_lib is not None:
        return True
    path = os.path.join(_ACCEL_ROOT, "go", "neurons", "dpi_neuron", "libdpi_neuron.so")
    if not os.path.isfile(path):
        return False
    try:
        library = ctypes.CDLL(path)
    except OSError:
        return False
    _go_lib = _configure_c_library(library, "dpi_neuron_simulate_c", mojo=False)
    _HAS_GO = _go_lib is not None
    return _HAS_GO


def ensure_mojo_loaded() -> bool:
    """Load the compiled DPI Mojo C ABI when available."""
    global _mojo_lib, _HAS_MOJO
    if _mojo_lib is not None:
        return True
    path = os.path.join(_ACCEL_ROOT, "mojo", "kernels", "libdpi_neuron.so")
    if not os.path.isfile(path):
        return False
    try:
        library = ctypes.CDLL(path)
    except OSError:
        return False
    _mojo_lib = _configure_c_library(library, "dpi_neuron_simulate_c", mojo=True)
    _HAS_MOJO = _mojo_lib is not None
    return _HAS_MOJO


def simulate_rust(n_steps: int, current: float) -> _DPIResult:
    """Run the factory-default Rust engine recurrence."""
    engine_class = _EngineDPICls
    if engine_class is None:
        raise RuntimeError("Rust DPI engine is unavailable.")
    neuron = engine_class()
    trace = np.empty(n_steps, dtype=np.float64)
    spikes = 0
    for index in range(n_steps):
        spikes += int(neuron.step(float(current)))
        trace[index] = float(neuron.get_state()["i_mem"])
    state = neuron.get_state()
    return (
        trace,
        spikes,
        (
            float(state["i_mem"]),
            float(state["i_ahp"]),
            float(state["refractory_time"]),
        ),
    )


def simulate_rust_complete(
    i_mem: float,
    i_ahp: float,
    refractory_time: float,
    i_threshold: float,
    i_reset: float,
    i_rest: float,
    i_tau: float,
    i_g: float,
    i_tau_ahp: float,
    i_ga: float,
    i_spike: float,
    i_0: float,
    kappa: float,
    alpha: float,
    tau: float,
    tau_ahp: float,
    refractory_period: float,
    dt: float,
    n_steps: int,
    current: float,
) -> _DPICompleteResult:
    """Run the complete configurable production Rust batch."""
    if _EngineDPICompleteFn is None:
        raise RuntimeError("Rust DPI complete batch is unavailable.")
    result = _EngineDPICompleteFn(
        i_mem,
        i_ahp,
        refractory_time,
        i_threshold,
        i_reset,
        i_rest,
        i_tau,
        i_g,
        i_tau_ahp,
        i_ga,
        i_spike,
        i_0,
        kappa,
        alpha,
        tau,
        tau_ahp,
        refractory_period,
        dt,
        n_steps,
        current,
    )
    raw_events = result[3]
    events: object
    if isinstance(raw_events, (bytes, bytearray, memoryview)):
        events = np.frombuffer(raw_events, dtype=np.uint8).copy()
    else:
        events = raw_events
    return (
        result[0],
        result[1],
        result[2],
        events,
        (
            float(result[4]),
            float(result[5]),
            float(result[6]),
        ),
    )


def simulate_julia(
    i_mem: float,
    i_ahp: float,
    refractory_time: float,
    i_threshold: float,
    i_reset: float,
    i_rest: float,
    i_tau: float,
    i_g: float,
    i_tau_ahp: float,
    i_ga: float,
    i_spike: float,
    i_0: float,
    kappa: float,
    alpha: float,
    tau: float,
    tau_ahp: float,
    refractory_period: float,
    dt: float,
    n_steps: int,
    current: float,
) -> _DPIResult:
    """Run the Julia recurrence with the complete circuit contract."""
    module = _julia_module
    if module is None:
        raise RuntimeError("Julia DPI module is unavailable.")
    result = module.simulate_trace(
        i_mem,
        i_ahp,
        refractory_time,
        i_threshold,
        i_reset,
        i_rest,
        i_tau,
        i_g,
        i_tau_ahp,
        i_ga,
        i_spike,
        i_0,
        kappa,
        alpha,
        tau,
        tau_ahp,
        refractory_period,
        dt,
        n_steps,
        current,
    )
    trace = np.ascontiguousarray(np.asarray(result.trace, dtype=np.float64))
    state = (float(result.i_mem_f), float(result.i_ahp_f), float(result.refractory_time_f))
    return trace, int(result.spikes), state


def simulate_julia_complete(
    i_mem: float,
    i_ahp: float,
    refractory_time: float,
    i_threshold: float,
    i_reset: float,
    i_rest: float,
    i_tau: float,
    i_g: float,
    i_tau_ahp: float,
    i_ga: float,
    i_spike: float,
    i_0: float,
    kappa: float,
    alpha: float,
    tau: float,
    tau_ahp: float,
    refractory_period: float,
    dt: float,
    n_steps: int,
    current: float,
) -> _DPICompleteResult:
    """Run the complete configurable Julia batch."""
    module = _julia_module
    if module is None:
        raise RuntimeError("Julia DPI module is unavailable.")
    result = module.simulate_complete(
        i_mem,
        i_ahp,
        refractory_time,
        i_threshold,
        i_reset,
        i_rest,
        i_tau,
        i_g,
        i_tau_ahp,
        i_ga,
        i_spike,
        i_0,
        kappa,
        alpha,
        tau,
        tau_ahp,
        refractory_period,
        dt,
        n_steps,
        current,
    )
    return (
        result.i_mem,
        result.i_ahp,
        result.refractory,
        result.events,
        (
            float(result.i_mem_f),
            float(result.i_ahp_f),
            float(result.refractory_time_f),
        ),
    )


def _simulate_c(
    library: Any,
    values: tuple[float, ...],
    n_steps: int,
    current: float,
    *,
    mojo: bool,
) -> _DPIResult:
    """Run a staged Go or Mojo DPI C ABI."""
    output = np.empty(n_steps + 3, dtype=np.float64)
    destination: Any
    if mojo:
        destination = int(output.ctypes.data)
    else:
        destination = output.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    spikes = int(library.dpi_neuron_simulate_c(*values, n_steps, current, destination))
    if spikes < 0:
        backend = "Mojo" if mojo else "Go"
        raise FloatingPointError(f"{backend} DPI kernel rejected the simulation contract.")
    state = (float(output[n_steps]), float(output[n_steps + 1]), float(output[n_steps + 2]))
    return np.ascontiguousarray(output[:n_steps]), spikes, state


def _simulate_c_complete(
    library: Any,
    values: tuple[float, ...],
    n_steps: int,
    current: float,
    *,
    mojo: bool,
) -> _DPICompleteResult:
    """Run one failure-atomic complete Go or Mojo C ABI packet."""
    i_mem = np.empty(n_steps + 1, dtype=np.float64)
    i_ahp = np.empty(n_steps + 1, dtype=np.float64)
    refractory = np.empty(n_steps + 1, dtype=np.float64)
    events = np.empty(n_steps, dtype=np.uint8)
    if mojo:
        destinations: tuple[Any, ...] = (
            int(i_mem.ctypes.data),
            int(i_ahp.ctypes.data),
            int(refractory.ctypes.data),
            int(events.ctypes.data),
        )
    else:
        destinations = (
            i_mem.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            i_ahp.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            refractory.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            events.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        )
    count = int(library.dpi_neuron_simulate_complete_c(*values, n_steps, current, *destinations))
    backend = "Mojo" if mojo else "Go"
    if count < 0 or count != int(np.sum(events, dtype=np.int64)):
        raise FloatingPointError(f"{backend} DPI complete kernel rejected its packet.")
    state = (float(i_mem[n_steps]), float(i_ahp[n_steps]), float(refractory[n_steps]))
    return (
        np.ascontiguousarray(i_mem[:n_steps]),
        np.ascontiguousarray(i_ahp[:n_steps]),
        np.ascontiguousarray(refractory[:n_steps]),
        np.ascontiguousarray(events),
        state,
    )


def simulate_go(
    i_mem: float,
    i_ahp: float,
    refractory_time: float,
    i_threshold: float,
    i_reset: float,
    i_rest: float,
    i_tau: float,
    i_g: float,
    i_tau_ahp: float,
    i_ga: float,
    i_spike: float,
    i_0: float,
    kappa: float,
    alpha: float,
    tau: float,
    tau_ahp: float,
    refractory_period: float,
    dt: float,
    n_steps: int,
    current: float,
) -> _DPIResult:
    """Run the Go recurrence through its C ABI."""
    if _go_lib is None:
        raise RuntimeError("Go DPI library is unavailable.")
    values = (
        i_mem,
        i_ahp,
        refractory_time,
        i_threshold,
        i_reset,
        i_rest,
        i_tau,
        i_g,
        i_tau_ahp,
        i_ga,
        i_spike,
        i_0,
        kappa,
        alpha,
        tau,
        tau_ahp,
        refractory_period,
        dt,
    )
    return _simulate_c(_go_lib, values, n_steps, current, mojo=False)


def simulate_mojo(
    i_mem: float,
    i_ahp: float,
    refractory_time: float,
    i_threshold: float,
    i_reset: float,
    i_rest: float,
    i_tau: float,
    i_g: float,
    i_tau_ahp: float,
    i_ga: float,
    i_spike: float,
    i_0: float,
    kappa: float,
    alpha: float,
    tau: float,
    tau_ahp: float,
    refractory_period: float,
    dt: float,
    n_steps: int,
    current: float,
) -> _DPIResult:
    """Run the Mojo recurrence through its C ABI."""
    if _mojo_lib is None:
        raise RuntimeError("Mojo DPI library is unavailable.")
    values = (
        i_mem,
        i_ahp,
        refractory_time,
        i_threshold,
        i_reset,
        i_rest,
        i_tau,
        i_g,
        i_tau_ahp,
        i_ga,
        i_spike,
        i_0,
        kappa,
        alpha,
        tau,
        tau_ahp,
        refractory_period,
        dt,
    )
    return _simulate_c(_mojo_lib, values, n_steps, current, mojo=True)


def simulate_go_complete(
    i_mem: float,
    i_ahp: float,
    refractory_time: float,
    i_threshold: float,
    i_reset: float,
    i_rest: float,
    i_tau: float,
    i_g: float,
    i_tau_ahp: float,
    i_ga: float,
    i_spike: float,
    i_0: float,
    kappa: float,
    alpha: float,
    tau: float,
    tau_ahp: float,
    refractory_period: float,
    dt: float,
    n_steps: int,
    current: float,
) -> _DPICompleteResult:
    """Run the complete configurable Go batch through its C ABI."""
    if _go_lib is None:
        raise RuntimeError("Go DPI library is unavailable.")
    values = (
        i_mem,
        i_ahp,
        refractory_time,
        i_threshold,
        i_reset,
        i_rest,
        i_tau,
        i_g,
        i_tau_ahp,
        i_ga,
        i_spike,
        i_0,
        kappa,
        alpha,
        tau,
        tau_ahp,
        refractory_period,
        dt,
    )
    return _simulate_c_complete(_go_lib, values, n_steps, current, mojo=False)


def simulate_mojo_complete(
    i_mem: float,
    i_ahp: float,
    refractory_time: float,
    i_threshold: float,
    i_reset: float,
    i_rest: float,
    i_tau: float,
    i_g: float,
    i_tau_ahp: float,
    i_ga: float,
    i_spike: float,
    i_0: float,
    kappa: float,
    alpha: float,
    tau: float,
    tau_ahp: float,
    refractory_period: float,
    dt: float,
    n_steps: int,
    current: float,
) -> _DPICompleteResult:
    """Run the complete configurable Mojo batch through its C ABI."""
    if _mojo_lib is None:
        raise RuntimeError("Mojo DPI library is unavailable.")
    values = (
        i_mem,
        i_ahp,
        refractory_time,
        i_threshold,
        i_reset,
        i_rest,
        i_tau,
        i_g,
        i_tau_ahp,
        i_ga,
        i_spike,
        i_0,
        kappa,
        alpha,
        tau,
        tau_ahp,
        refractory_period,
        dt,
    )
    return _simulate_c_complete(_mojo_lib, values, n_steps, current, mojo=True)
