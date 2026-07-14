# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Poisson acceleration backend loading and execution

"""Load full-contract seeded Rust, Julia, Go, and Mojo Poisson backends."""

from __future__ import annotations

import ctypes
import importlib
import importlib.util as importlib_util
import os
from typing import Any, Protocol, cast

import numpy as np
import numpy.typing as npt

PoissonResult = tuple[npt.NDArray[np.uint8], int]


class _EngineRunner(Protocol):
    """Typed PyO3 batch boundary."""

    def __call__(
        self,
        rate_hz: float,
        dt_ms: float,
        rng_state: int,
        n_steps: int,
        rate_override: float,
    ) -> tuple[npt.NDArray[np.uint8], int]: ...


def _load_engine_poisson() -> _EngineRunner:
    """Return the installed Rust engine's complete seeded batch function."""
    engine = importlib.import_module("sc_neurocore_engine")
    return cast(_EngineRunner, engine.py_poisson_simulate)


try:
    _engine_simulate: _EngineRunner | None = _load_engine_poisson()
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


def ensure_julia_loaded() -> bool:
    """Load the committed Julia module when ``juliacall`` is available.

    Returns
    -------
    bool
        ``True`` when the Julia recurrence is ready for execution, otherwise
        ``False``. Import, source, or runtime failures remain non-fatal probes.
    """
    global _julia_module, _HAS_JULIA
    if _julia_module is not None:
        return True
    if importlib_util.find_spec("juliacall") is None:
        return False
    source = os.path.join(_ACCEL_ROOT, "julia", "neurons", "poisson.jl")
    if not os.path.isfile(source):
        return False
    try:
        juliacall = importlib.import_module("juliacall")
        julia = juliacall.Main
        julia.include(source)
        _julia_module = julia.PoissonAccel
    except (ImportError, AttributeError, RuntimeError):
        return False
    _HAS_JULIA = True
    return True


def _configure_c_library(library: Any, *, mojo: bool) -> Any | None:
    """Bind the real Poisson C ABI for Go or Mojo."""
    simulate = getattr(library, "poisson_simulate_c", None)
    if simulate is None:
        return None
    rng_type = ctypes.c_int64 if mojo else ctypes.c_uint16
    destination = ctypes.c_int64 if mojo else ctypes.POINTER(ctypes.c_double)
    simulate.argtypes = [
        ctypes.c_double,
        ctypes.c_double,
        rng_type,
        ctypes.c_int64,
        ctypes.c_double,
        destination,
    ]
    simulate.restype = ctypes.c_int64
    return library


def ensure_go_loaded() -> bool:
    """Load the staged Go C-shared Poisson library.

    Returns
    -------
    bool
        ``True`` when the library exports the configured Poisson ABI, otherwise
        ``False``.
    """
    global _go_lib, _HAS_GO
    if _go_lib is not None:
        return True
    path = os.path.join(_ACCEL_ROOT, "go", "neurons", "poisson", "libpoisson.so")
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
    """Load the staged Mojo Poisson shared library.

    Returns
    -------
    bool
        ``True`` when the library exports the configured Poisson ABI, otherwise
        ``False``.
    """
    global _mojo_lib, _HAS_MOJO
    if _mojo_lib is not None:
        return True
    path = os.path.join(_ACCEL_ROOT, "mojo", "kernels", "libpoisson.so")
    if not os.path.isfile(path):
        return False
    try:
        library = ctypes.CDLL(path)
    except OSError:
        return False
    _mojo_lib = _configure_c_library(library, mojo=True)
    _HAS_MOJO = _mojo_lib is not None
    return _HAS_MOJO


def _normalise_result(events: npt.ArrayLike, final_rng: int | float) -> PoissonResult:
    """Validate and normalise one native event/RNG result."""
    try:
        event_values = np.asarray(events, dtype=np.float64)
        final_rng_value = float(final_rng)
    except (TypeError, ValueError, OverflowError) as exc:
        raise FloatingPointError("Poisson backend returned non-numeric state.") from exc
    if event_values.ndim != 1:
        raise FloatingPointError("Poisson backend returned a malformed event trace.")
    if not np.isfinite(event_values).all() or not np.isin(event_values, (0.0, 1.0)).all():
        raise FloatingPointError("Poisson backend returned non-binary events.")
    if (
        isinstance(final_rng, bool)
        or not np.isfinite(final_rng_value)
        or not final_rng_value.is_integer()
        or not 1 <= final_rng_value <= 0xFFFF
    ):
        raise FloatingPointError("Poisson backend returned an invalid LFSR state.")
    return np.ascontiguousarray(event_values, dtype=np.uint8), int(final_rng_value)


def simulate_rust(
    rate_hz: float,
    dt_ms: float,
    rng_state: int,
    n_steps: int,
    rate_override: float,
) -> PoissonResult:
    """Run the complete contract through the production Rust engine.

    Parameters
    ----------
    rate_hz : float
        Configured homogeneous rate in hertz.
    dt_ms : float
        Bin width in milliseconds.
    rng_state : int
        Non-zero 16-bit LFSR state at batch entry.
    n_steps : int
        Number of binary time bins to generate.
    rate_override : float
        Batch rate in hertz, or a negative value to select ``rate_hz``.

    Returns
    -------
    events : numpy.ndarray
        Validated contiguous ``uint8`` event trace.
    final_rng : int
        Non-zero 16-bit LFSR state at batch exit.

    Raises
    ------
    RuntimeError
        If the Rust engine boundary is unavailable.
    FloatingPointError
        If the engine returns malformed event or RNG data.
    """
    simulate = _engine_simulate
    if simulate is None:
        raise RuntimeError("Rust Poisson engine is unavailable.")
    return _normalise_result(*simulate(rate_hz, dt_ms, rng_state, n_steps, rate_override))


def simulate_julia(
    rate_hz: float,
    dt_ms: float,
    rng_state: int,
    n_steps: int,
    rate_override: float,
) -> PoissonResult:
    """Run the committed Julia recurrence with the complete contract.

    Parameters
    ----------
    rate_hz : float
        Configured homogeneous rate in hertz.
    dt_ms : float
        Bin width in milliseconds.
    rng_state : int
        Non-zero 16-bit LFSR state at batch entry.
    n_steps : int
        Number of binary time bins to generate.
    rate_override : float
        Batch rate in hertz, or a negative value to select ``rate_hz``.

    Returns
    -------
    events : numpy.ndarray
        Validated contiguous ``uint8`` event trace.
    final_rng : int
        Non-zero 16-bit LFSR state at batch exit.

    Raises
    ------
    RuntimeError
        If the Julia module is unavailable.
    FloatingPointError
        If the module returns malformed event or RNG data.
    """
    module = _julia_module
    if module is None:
        raise RuntimeError("Julia Poisson module is unavailable.")
    result = module.simulate_trace(rate_hz, dt_ms, rng_state, n_steps, rate_override)
    return _normalise_result(result.events, result.rng_state_f)


def _simulate_c(
    library: Any,
    rate_hz: float,
    dt_ms: float,
    rng_state: int,
    n_steps: int,
    rate_override: float,
    *,
    mojo: bool,
) -> PoissonResult:
    """Run one staged Go or Mojo C ABI with atomic output validation."""
    output = np.empty(n_steps + 1, dtype=np.float64)
    destination: Any
    if mojo:
        destination = int(output.ctypes.data)
    else:
        destination = output.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    spikes = int(
        library.poisson_simulate_c(
            rate_hz,
            dt_ms,
            rng_state,
            n_steps,
            rate_override,
            destination,
        )
    )
    if spikes < 0:
        backend = "Mojo" if mojo else "Go"
        raise FloatingPointError(f"{backend} Poisson kernel rejected the contract.")
    result = _normalise_result(output[:n_steps], output[n_steps])
    if spikes != int(np.sum(result[0], dtype=np.int64)):
        raise FloatingPointError("Poisson C ABI spike count disagrees with its event trace.")
    return result


def simulate_go(
    rate_hz: float,
    dt_ms: float,
    rng_state: int,
    n_steps: int,
    rate_override: float,
) -> PoissonResult:
    """Run the Go recurrence through its generated C ABI.

    Parameters
    ----------
    rate_hz : float
        Configured homogeneous rate in hertz.
    dt_ms : float
        Bin width in milliseconds.
    rng_state : int
        Non-zero 16-bit LFSR state at batch entry.
    n_steps : int
        Number of binary time bins to generate.
    rate_override : float
        Batch rate in hertz, or a negative value to select ``rate_hz``.

    Returns
    -------
    events : numpy.ndarray
        Validated contiguous ``uint8`` event trace.
    final_rng : int
        Non-zero 16-bit LFSR state at batch exit.

    Raises
    ------
    RuntimeError
        If the Go shared library is unavailable.
    FloatingPointError
        If the C ABI rejects the contract or returns inconsistent data.
    """
    if _go_lib is None:
        raise RuntimeError("Go Poisson library is unavailable.")
    return _simulate_c(
        _go_lib,
        rate_hz,
        dt_ms,
        rng_state,
        n_steps,
        rate_override,
        mojo=False,
    )


def simulate_mojo(
    rate_hz: float,
    dt_ms: float,
    rng_state: int,
    n_steps: int,
    rate_override: float,
) -> PoissonResult:
    """Run the Mojo recurrence through its shared-library ABI.

    Parameters
    ----------
    rate_hz : float
        Configured homogeneous rate in hertz.
    dt_ms : float
        Bin width in milliseconds.
    rng_state : int
        Non-zero 16-bit LFSR state at batch entry.
    n_steps : int
        Number of binary time bins to generate.
    rate_override : float
        Batch rate in hertz, or a negative value to select ``rate_hz``.

    Returns
    -------
    events : numpy.ndarray
        Validated contiguous ``uint8`` event trace.
    final_rng : int
        Non-zero 16-bit LFSR state at batch exit.

    Raises
    ------
    RuntimeError
        If the Mojo shared library is unavailable.
    FloatingPointError
        If the C ABI rejects the contract or returns inconsistent data.
    """
    if _mojo_lib is None:
        raise RuntimeError("Mojo Poisson library is unavailable.")
    return _simulate_c(
        _mojo_lib,
        rate_hz,
        dt_ms,
        rng_state,
        n_steps,
        rate_override,
        mojo=True,
    )
