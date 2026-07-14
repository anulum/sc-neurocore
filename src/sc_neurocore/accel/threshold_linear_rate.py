# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Threshold-linear acceleration loading and result validation

"""Load full-contract Rust, Julia, Go, and Mojo threshold-linear batches."""

from __future__ import annotations

import ctypes
import importlib
import importlib.util as importlib_util
import os
from typing import Any, Protocol, SupportsFloat, cast

import numpy as np
import numpy.typing as npt

from sc_neurocore.accel.backend_order import with_floor
from sc_neurocore.accel.backend_selection import select_backend_order

ThresholdLinearRateResult = tuple[npt.NDArray[np.float64], float]
KERNEL = "threshold_linear_rate_algebraic_batch"
_AUTO_BACKENDS = with_floor("python")


class _EngineRunner(Protocol):
    """Typed PyO3 threshold-linear batch boundary."""

    def __call__(
        self,
        r: float,
        theta: float,
        gain: float,
        n_steps: int,
        current: float,
    ) -> tuple[npt.NDArray[np.float64], float]: ...


def _load_engine_runner() -> _EngineRunner:
    """Return the installed Rust engine batch function."""
    engine = importlib.import_module("sc_neurocore_engine")
    return cast(_EngineRunner, engine.py_threshold_linear_rate_simulate)


try:
    _engine_simulate: _EngineRunner | None = _load_engine_runner()
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
    """Load the committed Julia module when ``juliacall`` is available."""
    global _julia_module, _HAS_JULIA
    if _julia_module is not None:
        return True
    if importlib_util.find_spec("juliacall") is None:
        return False
    source = os.path.join(_ACCEL_ROOT, "julia", "neurons", "threshold_linear_rate.jl")
    if not os.path.isfile(source):
        return False
    try:
        juliacall = importlib.import_module("juliacall")
        julia = juliacall.Main
        julia.include(source)
        _julia_module = julia.ThresholdLinearRateAccel
    except (ImportError, AttributeError, RuntimeError):
        return False
    _HAS_JULIA = True
    return True


def _configure_c_library(library: Any, *, mojo: bool) -> Any | None:
    """Bind one configurable threshold-linear C ABI."""
    simulate = getattr(library, "threshold_linear_rate_simulate_c", None)
    if simulate is None:
        return None
    destination = ctypes.c_int64 if mojo else ctypes.POINTER(ctypes.c_double)
    simulate.argtypes = [
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_int64,
        ctypes.c_double,
        destination,
    ]
    simulate.restype = ctypes.c_int64
    return library


def ensure_go_loaded() -> bool:
    """Load the staged Go threshold-linear C-shared library."""
    global _go_lib, _HAS_GO
    if _go_lib is not None:
        return True
    path = os.path.join(
        _ACCEL_ROOT,
        "go",
        "neurons",
        "threshold_linear_rate",
        "libthreshold_linear_rate.so",
    )
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
    """Load the staged Mojo threshold-linear shared library."""
    global _mojo_lib, _HAS_MOJO
    if _mojo_lib is not None:
        return True
    path = os.path.join(
        _ACCEL_ROOT,
        "mojo",
        "kernels",
        "libthreshold_linear_rate.so",
    )
    if not os.path.isfile(path):
        return False
    try:
        library = ctypes.CDLL(path)
    except OSError:
        return False
    _mojo_lib = _configure_c_library(library, mojo=True)
    _HAS_MOJO = _mojo_lib is not None
    return _HAS_MOJO


def backend_available(backend: str) -> bool:
    """Return whether one public execution lane is ready."""
    if backend == "rust":
        return _HAS_RUST and _engine_simulate is not None
    if backend == "julia":
        return ensure_julia_loaded()
    if backend == "go":
        return ensure_go_loaded()
    if backend == "mojo":
        return ensure_mojo_loaded()
    return backend == "python"


def auto_backend() -> str:
    """Choose the first available lane from committed measured evidence."""
    ordered = select_backend_order(KERNEL, static=_AUTO_BACKENDS)
    return next((backend for backend in ordered if backend_available(backend)), "python")


def normalise_result(
    trace: npt.ArrayLike,
    final_rate: object,
    *,
    n_steps: int,
    initial_rate: float,
) -> ThresholdLinearRateResult:
    """Reject malformed or non-atomic backend output before public commit."""
    try:
        values = np.asarray(trace, dtype=np.float64)
        final = float(cast(SupportsFloat, final_rate))
    except (TypeError, ValueError, OverflowError) as exc:
        raise FloatingPointError("ThresholdLinearRate backend returned non-numeric state.") from exc
    if values.ndim != 1 or values.shape != (n_steps,):
        raise FloatingPointError("ThresholdLinearRate backend returned a malformed rate trace.")
    if not np.isfinite(values).all() or not (values >= 0.0).all():
        raise FloatingPointError(
            "ThresholdLinearRate backend returned a non-finite or negative rate."
        )
    if not np.isfinite(final) or final < 0.0:
        raise FloatingPointError("ThresholdLinearRate backend returned an invalid final rate.")
    expected_final = initial_rate if n_steps == 0 else float(values[-1])
    if final != expected_final:
        raise FloatingPointError("ThresholdLinearRate backend final rate disagrees with its trace.")
    return np.ascontiguousarray(values, dtype=np.float64), final


def simulate_rust(
    r: float,
    theta: float,
    gain: float,
    n_steps: int,
    current: float,
) -> ThresholdLinearRateResult:
    """Run the complete contract through the production Rust engine."""
    if _engine_simulate is None:
        raise RuntimeError("Rust ThresholdLinearRate engine is unavailable.")
    return normalise_result(
        *_engine_simulate(r, theta, gain, n_steps, current),
        n_steps=n_steps,
        initial_rate=r,
    )


def simulate_julia(
    r: float,
    theta: float,
    gain: float,
    n_steps: int,
    current: float,
) -> ThresholdLinearRateResult:
    """Run the complete contract through the committed Julia module."""
    if _julia_module is None:
        raise RuntimeError("Julia ThresholdLinearRate module is unavailable.")
    result = _julia_module.simulate_trace(
        float(r),
        float(theta),
        float(gain),
        n_steps,
        float(current),
    )
    return normalise_result(
        result.trace,
        result.rf,
        n_steps=n_steps,
        initial_rate=r,
    )


def _simulate_c(
    library: Any,
    r: float,
    theta: float,
    gain: float,
    n_steps: int,
    current: float,
    *,
    mojo: bool,
) -> ThresholdLinearRateResult:
    """Run one staged C ABI and validate its complete output."""
    output = np.full(n_steps + 1, np.nan, dtype=np.float64)
    destination: Any = (
        int(output.ctypes.data) if mojo else output.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    )
    status = int(
        library.threshold_linear_rate_simulate_c(
            r,
            theta,
            gain,
            n_steps,
            current,
            destination,
        )
    )
    if status != 0:
        backend = "Mojo" if mojo else "Go"
        raise FloatingPointError(f"{backend} ThresholdLinearRate kernel rejected the contract.")
    return normalise_result(
        output[:n_steps],
        output[n_steps],
        n_steps=n_steps,
        initial_rate=r,
    )


def simulate_go(
    r: float,
    theta: float,
    gain: float,
    n_steps: int,
    current: float,
) -> ThresholdLinearRateResult:
    """Run the Go transfer through its generated C ABI."""
    if _go_lib is None:
        raise RuntimeError("Go ThresholdLinearRate library is unavailable.")
    return _simulate_c(_go_lib, r, theta, gain, n_steps, current, mojo=False)


def simulate_mojo(
    r: float,
    theta: float,
    gain: float,
    n_steps: int,
    current: float,
) -> ThresholdLinearRateResult:
    """Run the Mojo transfer through its exported C ABI."""
    if _mojo_lib is None:
        raise RuntimeError("Mojo ThresholdLinearRate library is unavailable.")
    return _simulate_c(_mojo_lib, r, theta, gain, n_steps, current, mojo=True)
