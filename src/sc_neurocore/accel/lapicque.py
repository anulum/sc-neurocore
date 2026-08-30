# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Lapicque acceleration backend loading and execution

"""Load and execute the native Lapicque acceleration backends."""

from __future__ import annotations

import ctypes
import importlib
import importlib.util as importlib_util
import os
from typing import Any, cast

import numpy as np
import numpy.typing as npt


def _load_engine_lapicque() -> tuple[type[Any], Any]:
    """Return the Rust compatibility class and complete-batch entrypoint."""
    engine = importlib.import_module("sc_neurocore_engine")
    return cast(type[Any], engine.LapicqueNeuron), engine.lapicque_simulate_complete


_EngineLapicqueCls: type[Any] | None
_EngineLapicqueCompleteFn: Any | None
try:
    _EngineLapicqueCls, _EngineLapicqueCompleteFn = _load_engine_lapicque()
    _HAS_RUST = True
except (ImportError, AttributeError):
    _EngineLapicqueCls = None
    _EngineLapicqueCompleteFn = None
    _HAS_RUST = False

_julia_module: Any | None = None
_HAS_JULIA = False
_go_lib: Any | None = None
_HAS_GO = False
_mojo_lib: Any | None = None
_HAS_MOJO = False

_ACCEL_ROOT = os.path.dirname(__file__)


def ensure_julia_loaded() -> bool:
    """Load the executable Lapicque Julia module when available."""
    global _julia_module, _HAS_JULIA
    if _julia_module is not None:
        return True
    if importlib_util.find_spec("juliacall") is None:
        return False
    source_path = os.path.join(_ACCEL_ROOT, "julia", "neurons", "lapicque.jl")
    if not os.path.isfile(source_path):
        return False
    try:
        juliacall = importlib.import_module("juliacall")
        julia = juliacall.Main
        julia.include(source_path)
        _julia_module = julia.LapicqueAccel
    except (ImportError, AttributeError, RuntimeError):
        return False
    _HAS_JULIA = True
    return True


def ensure_go_loaded() -> bool:
    """Load the compiled Lapicque Go C-ABI bridge when available."""
    global _go_lib, _HAS_GO
    if _go_lib is not None:
        return True
    library_path = os.path.join(_ACCEL_ROOT, "go", "neurons", "lapicque", "liblapicque.so")
    if not os.path.isfile(library_path):
        return False
    try:
        library = ctypes.CDLL(library_path)
    except OSError:
        return False
    simulate = getattr(library, "lapicque_simulate_c", None)
    complete = getattr(library, "lapicque_simulate_complete_c", None)
    if simulate is None or complete is None:
        return False
    simulate.argtypes = [ctypes.c_double] * 7 + [
        ctypes.c_int64,
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_double),
    ]
    simulate.restype = ctypes.c_int64
    complete.argtypes = [ctypes.c_double] * 10 + [
        ctypes.c_int64,
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
    """Load the compiled Lapicque Mojo C ABI when available."""
    global _mojo_lib, _HAS_MOJO
    if _mojo_lib is not None:
        return True
    library_path = os.path.join(_ACCEL_ROOT, "mojo", "kernels", "liblapicque.so")
    if not os.path.isfile(library_path):
        return False
    try:
        library = ctypes.CDLL(library_path)
    except OSError:
        return False
    simulate = getattr(library, "lapicque_simulate_c", None)
    complete = getattr(library, "lapicque_simulate_complete_c", None)
    if simulate is None or complete is None:
        return False
    simulate.argtypes = [ctypes.c_double] * 7 + [
        ctypes.c_int64,
        ctypes.c_double,
        ctypes.c_int64,
    ]
    simulate.restype = ctypes.c_int64
    complete.argtypes = [ctypes.c_double] * 10 + [
        ctypes.c_int64,
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
    """Run the factory-default Rust engine recurrence."""
    assert _EngineLapicqueCls is not None
    neuron = _EngineLapicqueCls()
    trace = np.empty(n_steps, dtype=np.float64)
    spikes = 0
    for index in range(n_steps):
        spikes += int(neuron.step(float(current)))
        trace[index] = float(neuron.get_state()["v"])
    return trace, spikes, float(neuron.get_state()["v"])


def simulate_rust_complete(
    v: float,
    v_rest: float,
    v_reset: float,
    v_threshold: float,
    tau: float,
    resistance: float,
    dt: float,
    capacitance: float,
    series_resistance: float,
    polarization_resistance: float,
    excited: bool,
    source_profile: bool,
    n_steps: int,
    drive: float,
) -> tuple[object, object, float, bool]:
    """Run the complete profile-explicit Rust production batch."""
    assert _EngineLapicqueCompleteFn is not None
    result = _EngineLapicqueCompleteFn(
        v,
        v_rest,
        v_reset,
        v_threshold,
        tau,
        resistance,
        dt,
        capacitance,
        series_resistance,
        polarization_resistance,
        excited,
        source_profile,
        n_steps,
        drive,
    )
    return result[0], result[1], float(result[2]), bool(result[3])


def simulate_julia(
    v: float,
    v_rest: float,
    v_reset: float,
    v_threshold: float,
    tau: float,
    resistance: float,
    dt: float,
    n_steps: int,
    current: float,
) -> tuple[npt.NDArray[np.float64], int, float]:
    """Run the Julia recurrence with the complete numeric contract."""
    assert _julia_module is not None
    result = _julia_module.simulate_trace(
        float(v),
        float(v_rest),
        float(v_reset),
        float(v_threshold),
        float(tau),
        float(resistance),
        float(dt),
        int(n_steps),
        float(current),
    )
    trace = np.ascontiguousarray(np.asarray(result.trace, dtype=np.float64))
    return trace, int(result.spikes), float(result.vf)


def simulate_julia_complete(
    v: float,
    v_rest: float,
    v_reset: float,
    v_threshold: float,
    tau: float,
    resistance: float,
    dt: float,
    capacitance: float,
    series_resistance: float,
    polarization_resistance: float,
    excited: bool,
    source_profile: bool,
    n_steps: int,
    drive: float,
) -> tuple[object, object, float, bool]:
    """Run the complete profile-explicit Julia batch."""
    assert _julia_module is not None
    result = _julia_module.simulate_complete(
        v,
        v_rest,
        v_reset,
        v_threshold,
        tau,
        resistance,
        dt,
        capacitance,
        series_resistance,
        polarization_resistance,
        excited,
        source_profile,
        n_steps,
        drive,
    )
    return result.trace, result.events, float(result.vf), bool(result.excited)


def simulate_go(
    v: float,
    v_rest: float,
    v_reset: float,
    v_threshold: float,
    tau: float,
    resistance: float,
    dt: float,
    n_steps: int,
    current: float,
) -> tuple[npt.NDArray[np.float64], int, float]:
    """Run the Go service recurrence through its C ABI."""
    assert _go_lib is not None
    output = np.empty(n_steps + 1, dtype=np.float64)
    spikes = int(
        _go_lib.lapicque_simulate_c(
            v,
            v_rest,
            v_reset,
            v_threshold,
            tau,
            resistance,
            dt,
            n_steps,
            current,
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
    )
    if spikes < 0:
        raise FloatingPointError("Go Lapicque kernel rejected the simulation contract.")
    return np.ascontiguousarray(output[:n_steps]), spikes, float(output[n_steps])


def simulate_go_complete(
    v: float,
    v_rest: float,
    v_reset: float,
    v_threshold: float,
    tau: float,
    resistance: float,
    dt: float,
    capacitance: float,
    series_resistance: float,
    polarization_resistance: float,
    excited: bool,
    source_profile: bool,
    n_steps: int,
    drive: float,
) -> tuple[object, object, float, bool]:
    """Run the complete Go batch through its mutation-free C ABI."""
    assert _go_lib is not None
    voltage = np.empty(n_steps + 1, dtype=np.float64)
    events = np.empty(n_steps, dtype=np.uint8)
    event_count = int(
        _go_lib.lapicque_simulate_complete_c(
            v,
            v_rest,
            v_reset,
            v_threshold,
            tau,
            resistance,
            dt,
            capacitance,
            series_resistance,
            polarization_resistance,
            int(excited),
            int(source_profile),
            n_steps,
            drive,
            voltage.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            events.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        )
    )
    if event_count < 0:
        raise FloatingPointError("Go Lapicque kernel rejected the simulation contract.")
    if event_count != int(np.sum(events, dtype=np.int64)):
        raise FloatingPointError("Go Lapicque event count disagrees with its event trace.")
    final_excited = bool(excited or (source_profile and event_count > 0))
    return voltage[:n_steps], events, float(voltage[n_steps]), final_excited


def simulate_mojo(
    v: float,
    v_rest: float,
    v_reset: float,
    v_threshold: float,
    tau: float,
    resistance: float,
    dt: float,
    n_steps: int,
    current: float,
) -> tuple[npt.NDArray[np.float64], int, float]:
    """Run the Mojo recurrence through its C ABI."""
    assert _mojo_lib is not None
    output = np.empty(n_steps + 1, dtype=np.float64)
    spikes = int(
        _mojo_lib.lapicque_simulate_c(
            v,
            v_rest,
            v_reset,
            v_threshold,
            tau,
            resistance,
            dt,
            n_steps,
            current,
            int(output.ctypes.data),
        )
    )
    if spikes < 0:
        raise FloatingPointError("Mojo Lapicque kernel rejected the simulation contract.")
    return np.ascontiguousarray(output[:n_steps]), spikes, float(output[n_steps])


def simulate_mojo_complete(
    v: float,
    v_rest: float,
    v_reset: float,
    v_threshold: float,
    tau: float,
    resistance: float,
    dt: float,
    capacitance: float,
    series_resistance: float,
    polarization_resistance: float,
    excited: bool,
    source_profile: bool,
    n_steps: int,
    drive: float,
) -> tuple[object, object, float, bool]:
    """Run the complete Mojo batch through its mutation-free C ABI."""
    assert _mojo_lib is not None
    voltage = np.empty(n_steps + 1, dtype=np.float64)
    events = np.empty(n_steps, dtype=np.uint8)
    event_count = int(
        _mojo_lib.lapicque_simulate_complete_c(
            v,
            v_rest,
            v_reset,
            v_threshold,
            tau,
            resistance,
            dt,
            capacitance,
            series_resistance,
            polarization_resistance,
            int(excited),
            int(source_profile),
            n_steps,
            drive,
            int(voltage.ctypes.data),
            int(events.ctypes.data),
        )
    )
    if event_count < 0:
        raise FloatingPointError("Mojo Lapicque kernel rejected the simulation contract.")
    if event_count != int(np.sum(events, dtype=np.int64)):
        raise FloatingPointError("Mojo Lapicque event count disagrees with its event trace.")
    final_excited = bool(excited or (source_profile and event_count > 0))
    return voltage[:n_steps], events, float(voltage[n_steps]), final_excited
