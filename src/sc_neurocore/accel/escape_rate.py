# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — EscapeRate acceleration backend loading and execution

"""Load full-contract seeded Rust, Julia, Go, and Mojo EscapeRate backends."""

from __future__ import annotations

import ctypes
import importlib
import importlib.util as importlib_util
import os
from typing import Any, Protocol, cast

import numpy as np
import numpy.typing as npt

EscapeResult = tuple[npt.NDArray[np.float64], npt.NDArray[np.uint8], float, int]


class _EngineRunner(Protocol):
    """Typed PyO3 batch boundary."""

    def __call__(
        self,
        v: float,
        v_rest: float,
        v_reset: float,
        v_threshold: float,
        tau_m: float,
        rho_0: float,
        delta_u: float,
        resistance: float,
        dt: float,
        rng_state: int,
        n_steps: int,
        current: float,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.uint8], float, int]: ...


def _load_engine_escape_rate() -> _EngineRunner:
    """Return the installed Rust engine's complete seeded batch function."""
    engine = importlib.import_module("sc_neurocore_engine")
    return cast(_EngineRunner, engine.py_escape_rate_simulate)


try:
    _engine_simulate: _EngineRunner | None = _load_engine_escape_rate()
    _HAS_RUST = True
except (ImportError, AttributeError):
    _engine_simulate = None
    _HAS_RUST = False

_julia_module: Any | None = None
_go_lib: Any | None = None
_mojo_lib: Any | None = None
_HAS_JULIA = False
_HAS_GO = False
_HAS_MOJO = False

_ACCEL_ROOT = os.path.dirname(__file__)
_DOUBLE_FIELDS = 9


def ensure_julia_loaded() -> bool:
    """Load the committed Julia module when ``juliacall`` is available."""
    global _julia_module, _HAS_JULIA
    if _julia_module is not None:
        return True
    if importlib_util.find_spec("juliacall") is None:
        return False
    source = os.path.join(_ACCEL_ROOT, "julia", "neurons", "escape_rate.jl")
    if not os.path.isfile(source):
        return False
    try:
        juliacall = importlib.import_module("juliacall")
        julia = juliacall.Main
        julia.include(source)
        _julia_module = julia.EscapeRateAccel
    except (ImportError, AttributeError, RuntimeError):
        return False
    _HAS_JULIA = True
    return True


def _configure_c_library(library: Any, *, mojo: bool) -> Any | None:
    """Bind the real EscapeRate C ABI for Go or Mojo."""
    simulate = getattr(library, "escape_rate_simulate_c", None)
    if simulate is None:
        return None
    rng_type = ctypes.c_int64 if mojo else ctypes.c_uint16
    destination = ctypes.c_int64 if mojo else ctypes.POINTER(ctypes.c_double)
    simulate.argtypes = [
        *([ctypes.c_double] * _DOUBLE_FIELDS),
        rng_type,
        ctypes.c_int64,
        ctypes.c_double,
        destination,
    ]
    simulate.restype = ctypes.c_int64
    return library


def ensure_go_loaded() -> bool:
    """Load the staged Go C-shared EscapeRate library."""
    global _go_lib, _HAS_GO
    if _go_lib is not None:
        return True
    path = os.path.join(_ACCEL_ROOT, "go", "neurons", "escape_rate", "libescape_rate.so")
    if not os.path.isfile(path):
        return False
    try:
        library = ctypes.CDLL(path)
    except OSError:
        return False
    _go_lib = _configure_c_library(library, mojo=False)
    _HAS_GO = _go_lib is not None
    return _HAS_GO


def ensure_mojo_loaded() -> bool:
    """Load the staged Mojo EscapeRate shared library."""
    global _mojo_lib, _HAS_MOJO
    if _mojo_lib is not None:
        return True
    path = os.path.join(_ACCEL_ROOT, "mojo", "kernels", "libescape_rate.so")
    if not os.path.isfile(path):
        return False
    try:
        library = ctypes.CDLL(path)
    except OSError:
        return False
    _mojo_lib = _configure_c_library(library, mojo=True)
    _HAS_MOJO = _mojo_lib is not None
    return _HAS_MOJO


def _normalise_result(
    trace: npt.ArrayLike,
    events: npt.ArrayLike,
    final_v: float,
    final_rng: int | float,
) -> EscapeResult:
    """Validate and normalise one native batch result."""
    try:
        trace_array = np.ascontiguousarray(np.asarray(trace, dtype=np.float64))
        event_values = np.asarray(events, dtype=np.float64)
        final_voltage = float(final_v)
        final_rng_value = float(final_rng)
    except (TypeError, ValueError, OverflowError) as exc:
        raise FloatingPointError("EscapeRate backend returned non-numeric state.") from exc
    if trace_array.shape != event_values.shape or trace_array.ndim != 1:
        raise FloatingPointError("EscapeRate backend returned malformed trace arrays.")
    if not np.isfinite(trace_array).all() or not np.isfinite(final_voltage):
        raise FloatingPointError("EscapeRate backend returned a non-finite state.")
    if not np.isfinite(event_values).all() or not np.isin(event_values, (0.0, 1.0)).all():
        raise FloatingPointError("EscapeRate backend returned non-binary events.")
    if (
        isinstance(final_rng, bool)
        or not np.isfinite(final_rng_value)
        or not final_rng_value.is_integer()
        or not 1 <= final_rng_value <= 0xFFFF
    ):
        raise FloatingPointError("EscapeRate backend returned an invalid LFSR state.")
    event_array = np.ascontiguousarray(event_values, dtype=np.uint8)
    return trace_array, event_array, final_voltage, int(final_rng_value)


def simulate_rust(
    v: float,
    v_rest: float,
    v_reset: float,
    v_threshold: float,
    tau_m: float,
    rho_0: float,
    delta_u: float,
    resistance: float,
    dt: float,
    rng_state: int,
    n_steps: int,
    current: float,
) -> EscapeResult:
    """Run the complete contract through the production Rust engine."""
    simulate = _engine_simulate
    if simulate is None:
        raise RuntimeError("Rust EscapeRate engine is unavailable.")
    return _normalise_result(
        *simulate(
            v,
            v_rest,
            v_reset,
            v_threshold,
            tau_m,
            rho_0,
            delta_u,
            resistance,
            dt,
            rng_state,
            n_steps,
            current,
        )
    )


def simulate_julia(
    v: float,
    v_rest: float,
    v_reset: float,
    v_threshold: float,
    tau_m: float,
    rho_0: float,
    delta_u: float,
    resistance: float,
    dt: float,
    rng_state: int,
    n_steps: int,
    current: float,
) -> EscapeResult:
    """Run the committed Julia recurrence with the complete contract."""
    module = _julia_module
    if module is None:
        raise RuntimeError("Julia EscapeRate module is unavailable.")
    result = module.simulate_trace(
        v,
        v_rest,
        v_reset,
        v_threshold,
        tau_m,
        rho_0,
        delta_u,
        resistance,
        dt,
        rng_state,
        n_steps,
        current,
    )
    return _normalise_result(result.trace, result.events, result.v_f, result.rng_state_f)


def _simulate_c(
    library: Any,
    values: tuple[float, ...],
    rng_state: int,
    n_steps: int,
    current: float,
    *,
    mojo: bool,
) -> EscapeResult:
    """Run one staged Go or Mojo C ABI with atomic output validation."""
    output = np.empty(2 * n_steps + 2, dtype=np.float64)
    destination: Any
    if mojo:
        destination = int(output.ctypes.data)
    else:
        destination = output.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    spikes = int(
        library.escape_rate_simulate_c(
            *values,
            rng_state,
            n_steps,
            current,
            destination,
        )
    )
    if spikes < 0:
        backend = "Mojo" if mojo else "Go"
        raise FloatingPointError(f"{backend} EscapeRate kernel rejected the contract.")
    events = output[n_steps : 2 * n_steps]
    result = _normalise_result(
        output[:n_steps],
        events,
        output[2 * n_steps],
        output[2 * n_steps + 1],
    )
    if spikes != int(np.sum(result[1], dtype=np.int64)):
        raise FloatingPointError("EscapeRate C ABI spike count disagrees with its event trace.")
    return result


def _values(
    v: float,
    v_rest: float,
    v_reset: float,
    v_threshold: float,
    tau_m: float,
    rho_0: float,
    delta_u: float,
    resistance: float,
    dt: float,
) -> tuple[float, ...]:
    return (v, v_rest, v_reset, v_threshold, tau_m, rho_0, delta_u, resistance, dt)


def simulate_go(
    v: float,
    v_rest: float,
    v_reset: float,
    v_threshold: float,
    tau_m: float,
    rho_0: float,
    delta_u: float,
    resistance: float,
    dt: float,
    rng_state: int,
    n_steps: int,
    current: float,
) -> EscapeResult:
    """Run the Go recurrence through its generated C ABI."""
    if _go_lib is None:
        raise RuntimeError("Go EscapeRate library is unavailable.")
    return _simulate_c(
        _go_lib,
        _values(v, v_rest, v_reset, v_threshold, tau_m, rho_0, delta_u, resistance, dt),
        rng_state,
        n_steps,
        current,
        mojo=False,
    )


def simulate_mojo(
    v: float,
    v_rest: float,
    v_reset: float,
    v_threshold: float,
    tau_m: float,
    rho_0: float,
    delta_u: float,
    resistance: float,
    dt: float,
    rng_state: int,
    n_steps: int,
    current: float,
) -> EscapeResult:
    """Run the Mojo recurrence through its shared-library ABI."""
    if _mojo_lib is None:
        raise RuntimeError("Mojo EscapeRate library is unavailable.")
    return _simulate_c(
        _mojo_lib,
        _values(v, v_rest, v_reset, v_threshold, tau_m, rho_0, delta_u, resistance, dt),
        rng_state,
        n_steps,
        current,
        mojo=True,
    )
